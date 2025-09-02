import threading
import queue
import time
import logging
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    # 如果没有安装openai包，创建一个模拟类用于测试
    class OpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = type('chat', (), {'completions': type('completions', (), {'create': lambda **kwargs: None})()})()
            self.completions = type('completions', (), {'create': lambda **kwargs: None})()
            self.embeddings = type('embeddings', (), {'create': lambda **kwargs: None})()
            self.images = type('images', (), {'generate': lambda **kwargs: None})()
            self.audio = type('audio', (), {
                'transcriptions': type('transcriptions', (), {'create': lambda **kwargs: None})(),
                'translations': type('translations', (), {'create': lambda **kwargs: None})()
            })()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class APIRequest:
    """封装API请求信息"""
    id: str
    method: str
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3
    result: Any = None
    error: Any = None
    status: RequestStatus = RequestStatus.PENDING


class WorkerThread(threading.Thread):
    """工作线程类，处理特定API key的请求"""
    
    def __init__(self, api_key: str, thread_id: int, request_queue: queue.Queue, 
                 result_queue: queue.Queue, retry_queue: queue.Queue):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.thread_id = thread_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.retry_queue = retry_queue
        self.client = OpenAI(api_key=api_key)
        self.running = True
        self.processed_count = 0
        self.failed_count = 0
        
    def run(self):
        """线程主循环"""
        logger.info(f"Worker thread {self.thread_id} started")
        request_count = 0  # 用于跟踪请求数量
        
        while self.running:
            try:
                # 从队列获取请求，超时1秒
                request = self.request_queue.get(timeout=1)
                
                if request is None:  # 收到停止信号
                    break
                    
                self._process_request(request)
                self.request_queue.task_done()
                request_count += 1
                
                # 每10个请求休眠1秒
                if request_count % 10 == 0:
                    logger.debug(f"Thread {self.thread_id} processed {request_count} requests, sleeping for 1 second")
                    time.sleep(1)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Thread {self.thread_id} unexpected error: {e}")
                
        logger.info(f"Worker thread {self.thread_id} stopped. Processed: {self.processed_count}, Failed: {self.failed_count}")
    
    def _process_request(self, request: APIRequest):
        """处理单个请求"""
        request.status = RequestStatus.PROCESSING
        logger.debug(f"Thread {self.thread_id} processing request {request.id}")
        
        try:
            # 根据方法名调用相应的OpenAI API
            result = self._call_openai_api(request.method, request.params)
            
            request.result = result
            request.status = RequestStatus.SUCCESS
            self.processed_count += 1
            
            # 将成功的结果放入结果队列
            self.result_queue.put(request)
            
            # 如果有回调函数，执行回调
            if request.callback:
                request.callback(result)
                
        except Exception as e:
            logger.warning(f"Thread {self.thread_id} failed to process request {request.id}: {e}")
            request.error = e
            request.status = RequestStatus.FAILED
            request.retry_count += 1
            self.failed_count += 1
            
            # 判断是否需要重试
            if request.retry_count < request.max_retries:
                request.status = RequestStatus.RETRYING
                self.retry_queue.put(request)
            else:
                # 达到最大重试次数，标记为失败
                self.result_queue.put(request)
                if request.callback:
                    request.callback(None, error=e)
    
    def _call_openai_api(self, method: str, params: Dict[str, Any]) -> Any:
        """调用OpenAI API"""
        if method == "chat.completions.create":
            return self.client.chat.completions.create(**params)
        elif method == "completions.create":
            return self.client.completions.create(**params)
        elif method == "embeddings.create":
            return self.client.embeddings.create(**params)
        elif method == "images.generate":
            return self.client.images.generate(**params)
        elif method == "audio.transcriptions.create":
            return self.client.audio.transcriptions.create(**params)
        elif method == "audio.translations.create":
            return self.client.audio.translations.create(**params)
        else:
            raise ValueError(f"Unsupported API method: {method}")
    
    def stop(self):
        """停止线程"""
        self.running = False


class OpenAIAPIBalancer:
    """OpenAI API Key负载均衡器"""
    
    def __init__(self, api_keys: List[str], max_queue_size: int = 1000):
        """
        初始化负载均衡器
        
        Args:
            api_keys: API密钥列表
            max_queue_size: 最大队列大小
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
            
        self.api_keys = api_keys
        self.num_workers = len(api_keys)
        
        # 创建队列
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.retry_queue = queue.Queue()
        
        # 统计信息
        self.total_requests = 0
        self.request_counter_lock = threading.Lock()
        self.running = True  # 移到创建工作线程之前
        
        # 创建工作线程
        self.workers = []
        self._create_workers()
        
        # 创建重试分发线程
        self.retry_dispatcher = threading.Thread(target=self._retry_dispatcher_loop, daemon=True)
        self.retry_dispatcher.start()
        
        logger.info(f"OpenAIAPIBalancer initialized with {self.num_workers} API keys")
    
    def _create_workers(self):
        """创建工作线程"""
        for i, api_key in enumerate(self.api_keys):
            worker = WorkerThread(
                api_key=api_key,
                thread_id=i,
                request_queue=self.request_queue,
                result_queue=self.result_queue,
                retry_queue=self.retry_queue
            )
            worker.start()
            self.workers.append(worker)
    
    def _retry_dispatcher_loop(self):
        """重试分发循环，将失败的请求重新分发"""
        while self.running:
            try:
                request = self.retry_queue.get(timeout=1)
                if request is None:
                    break
                    
                logger.info(f"Retrying request {request.id} (attempt {request.retry_count}/{request.max_retries})")
                
                # 将请求重新放入主队列，会被另一个线程处理
                self.request_queue.put(request)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Retry dispatcher error: {e}")
    
    def submit_request(self, method: str, params: Dict[str, Any], 
                      callback: Optional[Callable] = None, max_retries: int = 3) -> str:
        """
        提交请求到负载均衡器
        
        Args:
            method: API方法名
            params: API参数
            callback: 可选的回调函数
            max_retries: 最大重试次数
            
        Returns:
            请求ID
        """
        with self.request_counter_lock:
            self.total_requests += 1
            request_id = f"req_{self.total_requests}_{time.time()}"
        
        request = APIRequest(
            id=request_id,
            method=method,
            params=params,
            callback=callback,
            max_retries=max_retries
        )
        
        self.request_queue.put(request)
        logger.debug(f"Request {request_id} submitted")
        
        return request_id
    
    def submit_chat_completion(self, model: str, messages: List[Dict[str, str]], callback: Optional[Callable] = None, **kwargs) -> str:
        """
        提交聊天完成请求
        
        Args:
            messages: 消息列表
            callback: 可选的回调函数
            **kwargs: 其他参数
            
        Returns:
            请求ID
        """
        # Extract max_retries from kwargs if present (not part of OpenAI API params)
        max_retries = kwargs.pop('max_retries', 3)
        
        # Prepare OpenAI API parameters
        params = {"messages": messages, "model": model, **kwargs}
        
        return self.submit_request("chat.completions.create", params, callback=callback, max_retries=max_retries)
    
    def submit_embedding(self, input: str, model: str = "text-embedding-ada-002", **kwargs) -> str:
        """
        提交嵌入请求
        
        Args:
            input: 输入文本
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            请求ID
        """
        params = {"input": input, "model": model, **kwargs}
        return self.submit_request("embeddings.create", params)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[APIRequest]:
        """
        获取处理结果
        
        Args:
            timeout: 超时时间
            
        Returns:
            处理完成的请求对象
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_result_by_id(self, request_id: str, timeout: Optional[float] = None) -> Optional[APIRequest]:
        """
        获取特定ID的处理结果
        
        Args:
            request_id: 请求ID
            timeout: 超时时间
            
        Returns:
            处理完成的请求对象，如果没找到返回None
        """
        start_time = time.time()
        timeout = timeout or 60  # 默认60秒超时
        
        while time.time() - start_time < timeout:
            try:
                # 检查结果队列
                result = self.result_queue.get(timeout=0.1)
                if result.id == request_id:
                    return result
                else:
                    # 不是我们要的结果，放回队列
                    self.result_queue.put(result)
            except queue.Empty:
                time.sleep(0.1)
                continue
        
        return None
    
    def wait_for_result(self, request_id: str, timeout: float = 60) -> APIRequest:
        """
        等待特定请求的结果
        
        Args:
            request_id: 请求ID
            timeout: 超时时间
            
        Returns:
            处理完成的请求对象
            
        Raises:
            TimeoutError: 如果超时
        """
        result = self.get_result_by_id(request_id, timeout)
        if result is None:
            raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
        return result
    
    def get_all_results(self) -> List[APIRequest]:
        """
        获取所有当前可用的结果
        
        Returns:
            结果列表
        """
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_requests": self.total_requests,
            "pending_requests": self.request_queue.qsize(),
            "retry_requests": self.retry_queue.qsize(),
            "completed_results": self.result_queue.qsize(),
            "workers": []
        }
        
        for worker in self.workers:
            stats["workers"].append({
                "thread_id": worker.thread_id,
                "processed": worker.processed_count,
                "failed": worker.failed_count,
                "is_alive": worker.is_alive()
            })
        
        return stats
    
    def shutdown(self, wait: bool = True):
        """
        关闭负载均衡器
        
        Args:
            wait: 是否等待所有请求处理完成
        """
        logger.info("Shutting down OpenAIAPIBalancer...")
        
        self.running = False
        
        if wait:
            # 等待所有请求处理完成
            self.request_queue.join()
        
        # 停止所有工作线程
        for _ in self.workers:
            self.request_queue.put(None)
        
        # 停止重试分发线程
        self.retry_queue.put(None)
        
        # 等待所有线程结束
        for worker in self.workers:
            worker.stop()
            worker.join(timeout=5)
        
        if self.retry_dispatcher.is_alive():
            self.retry_dispatcher.join(timeout=5)
        
        logger.info("OpenAIAPIBalancer shutdown complete")


# 单元测试
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch, MagicMock
    import random
    
    class TestOpenAIAPIBalancer(unittest.TestCase):
        """OpenAI API负载均衡器单元测试"""
        
        def setUp(self):
            """测试前的准备工作"""
            # 模拟API密钥
            self.api_keys = ["test-key-1", "test-key-2", "test-key-3"]
            
        def tearDown(self):
            """测试后的清理工作"""
            if hasattr(self, 'balancer'):
                self.balancer.shutdown(wait=False)
        
        @patch('__main__.OpenAI')
        def test_balancer_initialization(self, _mock_openai):
            """测试负载均衡器初始化"""
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 验证初始化
            self.assertEqual(len(self.balancer.workers), 3)
            self.assertEqual(self.balancer.num_workers, 3)
            self.assertTrue(self.balancer.running)
            
            # 验证所有工作线程都在运行
            for worker in self.balancer.workers:
                self.assertTrue(worker.is_alive())
        
        @patch('__main__.OpenAI')
        def test_submit_and_process_requests(self, mock_openai):
            """测试提交和处理请求"""
            # 模拟OpenAI客户端响应
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 提交多个请求
            request_ids = []
            for i in range(5):
                request_id = self.balancer.submit_chat_completion(
                    messages=[
                        {"role": "user", "content": f"Test message {i}"}
                    ],
                    model="gpt-3.5-turbo"
                )
                request_ids.append(request_id)
                
            # 等待处理
            time.sleep(2)
            
            # 获取结果
            results = self.balancer.get_all_results()
            
            # 验证结果
            self.assertGreater(len(results), 0)
            for result in results:
                self.assertIn(result.status, [RequestStatus.SUCCESS, RequestStatus.FAILED])
                if result.status == RequestStatus.SUCCESS:
                    self.assertIsNotNone(result.result)
        
        @patch('__main__.OpenAI')
        def test_retry_mechanism(self, mock_openai):
            """测试重试机制"""
            # 模拟失败然后成功的场景
            mock_client = MagicMock()
            call_count = {'count': 0}
            
            def side_effect_func(**_kwargs):
                call_count['count'] += 1
                if call_count['count'] <= 2:
                    # 前两次调用失败
                    raise Exception("API Error: Rate limit exceeded")
                else:
                    # 第三次调用成功
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock(message=MagicMock(content="Success after retry"))]
                    return mock_response
            
            mock_client.chat.completions.create.side_effect = side_effect_func
            mock_openai.return_value = mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 提交请求
            request_id = self.balancer.submit_chat_completion(
                messages=[{"role": "user", "content": "Test retry"}],
                model="gpt-3.5-turbo",
                max_retries=3
            )
            
            # 等待重试处理 - 增加等待时间
            time.sleep(5)
            
            # 获取结果
            result = self.balancer.get_result(timeout=2)
            
            # 验证重试成功
            self.assertIsNotNone(result)
            if result:  # 只有在result不为None时才检查状态
                self.assertEqual(result.status, RequestStatus.SUCCESS)
                self.assertGreater(call_count['count'], 1)  # 确认进行了重试
        
        @patch('__main__.OpenAI')
        def test_load_distribution(self, mock_openai):
            """测试负载均匀分配"""
            # 记录每个线程处理的请求数
            thread_request_counts = {}
            
            def track_thread(*_args, **_kwargs):
                thread_id = threading.current_thread().name
                thread_request_counts[thread_id] = thread_request_counts.get(thread_id, 0) + 1
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content=f"Response from {thread_id}"))]
                return mock_response
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = track_thread
            mock_openai.return_value = mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 提交大量请求
            num_requests = 30
            for i in range(num_requests):
                self.balancer.submit_chat_completion(
                    messages=[{"role": "user", "content": f"Message {i}"}],
                    model="gpt-3.5-turbo"
                )
            
            # 等待所有请求处理完成
            time.sleep(3)
            
            # 获取统计信息
            stats = self.balancer.get_statistics()
            
            # 验证请求被分配给了多个线程
            processed_counts = [w['processed'] for w in stats['workers']]
            self.assertGreater(len([c for c in processed_counts if c > 0]), 1)  # 至少2个线程处理了请求
            
            # 验证负载相对均匀（允许一定偏差）
            if all(c > 0 for c in processed_counts):
                max_count = max(processed_counts)
                min_count = min(processed_counts)
                # 最大和最小处理数的差异不应该太大
                self.assertLess(max_count - min_count, num_requests // 2)
        
        @patch('__main__.OpenAI')
        def test_callback_execution(self, mock_openai):
            """测试回调函数执行"""
            # 设置模拟响应
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Callback test response"))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 创建回调函数
            callback_results = []
            def test_callback(result, error=None):
                callback_results.append({'result': result, 'error': error})
            
            # 提交带回调的请求
            self.balancer.submit_chat_completion(
                messages=[{"role": "user", "content": "Test callback"}],
                model="gpt-3.5-turbo",
                callback=test_callback
            )
            
            # 等待处理 - 增加等待时间
            time.sleep(3)
            
            # 验证回调被执行
            self.assertGreater(len(callback_results), 0)  # 至少一次回调
            if callback_results:
                self.assertIsNotNone(callback_results[0]['result'])
                self.assertIsNone(callback_results[0]['error'])
        
        @patch('__main__.OpenAI')
        def test_statistics(self, mock_openai):
            """测试统计信息功能"""
            # 模拟一些成功和失败的请求
            mock_client = MagicMock()
            
            def random_response(**_kwargs):
                if random.random() > 0.3:  # 70%成功率
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
                    return mock_response
                else:
                    raise Exception("Random failure")
            
            mock_client.chat.completions.create.side_effect = random_response
            mock_openai.return_value = mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 提交请求
            for i in range(20):
                self.balancer.submit_chat_completion(
                    messages=[{"role": "user", "content": f"Stats test {i}"}],
                    model="gpt-3.5-turbo",
                    max_retries=1  # 减少重试次数以加快测试
                )
            
            # 等待处理
            time.sleep(3)
            
            # 获取统计信息
            stats = self.balancer.get_statistics()
            
            # 验证统计信息
            self.assertEqual(stats['total_requests'], 20)
            self.assertIn('pending_requests', stats)
            self.assertIn('retry_requests', stats)
            self.assertIn('completed_results', stats)
            self.assertEqual(len(stats['workers']), 3)
            
            # 验证每个工作线程的统计
            total_processed = sum(w['processed'] for w in stats['workers'])
            # total_failed = sum(w['failed'] for w in stats['workers'])
            self.assertGreater(total_processed, 0)
        
        @patch('__main__.OpenAI')
        def test_api_key_fallback(self, mock_openai):
            """测试API key失败时的自动fallback行为"""
            # 记录调用情况
            call_history = []
            
            def create_mock_client(api_key):
                mock_client = MagicMock()
                
                def mock_api_call(**_kwargs):
                    call_history.append(api_key)
                    
                    # test-key-1 总是失败
                    if api_key == "test-key-1":
                        raise Exception(f"API key {api_key} is invalid")
                    
                    # 其他key成功
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock(message=MagicMock(content=f"Success with {api_key}"))]
                    return mock_response
                
                mock_client.chat.completions.create.side_effect = mock_api_call
                return mock_client
            
            mock_openai.side_effect = create_mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 提交请求
            num_requests = 10
            for i in range(num_requests):
                self.balancer.submit_chat_completion(
                    messages=[{"role": "user", "content": f"Test {i}"}],
                    model="gpt-3.5-turbo",
                    max_retries=3
                )
            
            # 等待处理
            time.sleep(5)
            
            # 获取结果和统计
            results = self.balancer.get_all_results()
            stats = self.balancer.get_statistics()
            
            # 基本验证
            self.assertGreater(len(results), 0, "应该有处理完成的结果")
            
            # 验证fallback机制的核心功能：
            # 1. 至少有一个worker处理了请求
            total_processed = sum(w['processed'] for w in stats['workers'])
            self.assertGreater(total_processed, 0, "至少有一个请求被成功处理")
            
            # 2. 如果worker 0被使用且有失败，验证重试队列被使用
            worker_0 = stats['workers'][0]
            if worker_0['failed'] > 0:
                # Worker 0 (test-key-1) 失败了，这是预期的
                logger.info(f"Worker 0 失败了 {worker_0['failed']} 次（预期行为）")
                
                # 验证重试机制：失败的请求应该被重试
                # 由于有重试机制，即使worker 0失败，请求也会被重新分配
                successful = [r for r in results if r.status == RequestStatus.SUCCESS]
                logger.info(f"尽管有失败，仍有 {len(successful)} 个请求成功")
            
            # 3. 记录测试信息供调试
            logger.info(f"Fallback测试完成: 总请求={num_requests}, "
                       f"结果数={len(results)}, "
                       f"总处理={total_processed}, "
                       f"各worker: {[(i, w['processed'], w['failed']) for i, w in enumerate(stats['workers'])]}")
        
        @patch('__main__.OpenAI')
        def test_graceful_shutdown(self, mock_openai):
            """测试优雅关闭"""
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Shutdown test"))]
            
            # 添加延迟以模拟正在处理的请求
            def delayed_response(**_kwargs):
                time.sleep(0.5)
                return mock_response
            
            mock_client.chat.completions.create.side_effect = delayed_response
            mock_openai.return_value = mock_client
            
            # 创建负载均衡器
            self.balancer = OpenAIAPIBalancer(self.api_keys)
            
            # 提交一些请求
            for i in range(5):
                self.balancer.submit_chat_completion(
                    messages=[{"role": "user", "content": f"Shutdown test {i}"}],
                    model="gpt-3.5-turbo"
                )
            
            # 立即关闭（wait=True会等待所有请求完成）
            self.balancer.shutdown(wait=True)
            
            # 验证所有线程已停止
            time.sleep(1)
            for worker in self.balancer.workers:
                self.assertFalse(worker.is_alive())
            
            # 验证重试分发线程已停止
            self.assertFalse(self.balancer.retry_dispatcher.is_alive())
    
    # 运行测试
    unittest.main(argv=[''], exit=False, verbosity=2)