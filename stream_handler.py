import asyncio
import json
import logging
from typing import AsyncGenerator, AsyncIterable, Callable, Optional, Tuple, Union

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class LLaMAStreamHandler:
    """
    Handles streaming responses from a LLaMA model, providing functionality for:
    - Token-by-token generation
    - SSE (Server-Sent Events) formatting
    - WebSocket support
    - Backpressure handling
    - Automatic chunk sizing based on network conditions and GPU generation speed.
    """

    def __init__(self, websocket: Optional[WebSocket] = None, initial_queue_size: int = 16):
        """
        Initializes the LLaMAStreamHandler.

        Args:
            websocket (Optional[WebSocket]): An optional WebSocket connection for sending data.
            initial_queue_size (int): Initial size of the queue for buffering tokens.
        """
        self.websocket = websocket
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=initial_queue_size)
        self.is_done = False
        self.token_count = 0
        self.error_occurred = False
        self.chunk_size = 1  # Initial chunk size, will be dynamically adjusted
        self.last_queue_size = initial_queue_size
        self.queue_full_count = 0
        self.consecutive_empty_reads = 0
        self.max_consecutive_empty_reads = 5
        self.min_chunk_size = 1
        self.max_chunk_size = 10
        self.adapt_chunk_size = True  # Enable dynamic chunk sizing by default

    async def stream_generate(self, generate_func: Callable[[], AsyncGenerator[str, None]]):
        """
        Streams tokens from the generator function into the queue.

        Args:
            generate_func (Callable[[], AsyncGenerator[str, None]]): An async generator function
                that yields tokens from the LLaMA model.
        """
        try:
            async for token in generate_func():
                await self.queue.put(token)
                self.token_count += 1
            await self.queue.put(None)  # Signal end of stream
            self.is_done = True
        except Exception as e:
            logger.exception("Error during token generation:")
            self.error_occurred = True
            await self.queue.put(None)  # Signal end of stream
            self.is_done = True

    async def format_sse_chunk(self) -> str:
        """
        Formats a chunk of tokens into Server-Sent Events (SSE) format.

        Returns:
            str: A string containing the SSE formatted chunk.  Returns an empty string if queue is empty.
        """
        data = ""
        tokens = []
        for _ in range(self.chunk_size):
            try:
                token = await self.queue.get()
                self.queue.task_done()
                if token is None:
                    self.is_done = True
                    break  # End of stream
                tokens.append(token)
            except asyncio.CancelledError:
                logger.warning("SSE formatting cancelled.")
                return ""
            except Exception as e:
                logger.error(f"Error getting token from queue: {e}")
                return ""

        if not tokens:
            self.consecutive_empty_reads += 1
            if self.consecutive_empty_reads > self.max_consecutive_empty_reads:
                logger.warning("Too many consecutive empty reads, stopping stream.")
                self.is_done = True
            return ""

        self.consecutive_empty_reads = 0
        data = "".join(tokens)
        sse_data = f"data: {json.dumps({'token': data})}\n\n"
        return sse_data

    async def handle_websocket(self) -> AsyncGenerator[str, None]:
        """
        Handles sending data over a WebSocket connection.

        Yields:
            str: Individual tokens received from the queue.
        """
        try:
            while not self.is_done or not self.queue.empty():
                token = await self.queue.get()
                self.queue.task_done()

                if token is None:
                    self.is_done = True
                    break

                if self.websocket:
                    try:
                        await self.websocket.send_text(token)
                    except Exception as e:
                        logger.error(f"Websocket send error: {e}")
                        self.is_done = True
                        break
                yield token  # Yield the token for any other processing

        except asyncio.CancelledError:
            logger.warning("WebSocket handling cancelled.")
        except Exception as e:
            logger.exception(f"Error in websocket handling: {e}")
            self.error_occurred = True
        finally:
            if self.websocket and self.websocket.client_state != 3: # 3 = WebSocketState.CLOSED
                await self.websocket.close()

    def buffer_management(self):
        """
        Dynamically adjusts the chunk size based on queue fullness.
        """
        if not self.adapt_chunk_size:
            return  # Skip dynamic chunk sizing

        current_queue_size = self.queue.maxsize - self.queue.qsize()
        queue_full_threshold = self.queue.maxsize * 0.8  # Consider queue "full" at 80% capacity

        if current_queue_size >= queue_full_threshold:
            self.queue_full_count += 1
            if self.queue_full_count > 3 and self.chunk_size > self.min_chunk_size:
                self.chunk_size = max(self.min_chunk_size, self.chunk_size - 1)  # Decrease chunk size
                self.queue_full_count = 0
                logger.debug(f"Decreasing chunk size to {self.chunk_size} due to queue fullness.")
        else:
            if self.chunk_size < self.max_chunk_size:
                self.chunk_size = min(self.max_chunk_size, self.chunk_size + 1)  # Increase chunk size
                logger.debug(f"Increasing chunk size to {self.chunk_size}.")
            self.queue_full_count = 0

        if self.last_queue_size != self.queue.maxsize:
            logger.warning(f"Queue size changed.  Was {self.last_queue_size}, now {self.queue.maxsize}. Dynamic chunk sizing may not work correctly.")
            self.last_queue_size = self.queue.maxsize