#!/usr/bin/env python3
"""
WebSocket Service for LLaMA Gateway
===================================

Provides real-time WebSocket communication for chat.
Enables live streaming and real-time interactions.

Author: EGO Revolution Team
Version: 1.0.0 - Phase 3 Advanced Features
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """Represents a WebSocket connection"""
    
    def __init__(self, websocket, connection_id: str, user_info: Dict[str, Any] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_info = user_info or {}
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.message_count = 0
        self.is_authenticated = False
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send a message to the client"""
        try:
            await self.websocket.send_text(json.dumps(message))
            self.last_activity = datetime.now()
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"❌ Error sending message to {self.connection_id}: {e}")
            return False
    
    async def send_error(self, error_message: str, error_code: str = "error") -> bool:
        """Send an error message to the client"""
        return await self.send_message({
            "type": "error",
            "error": error_message,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_success(self, message: str, data: Any = None) -> bool:
        """Send a success message to the client"""
        return await self.send_message({
            "type": "success",
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

class WebSocketService:
    """Service for managing WebSocket connections"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.rooms: Dict[str, Set[str]] = {}  # room_id -> set of connection_ids
        self.message_history: List[Dict[str, Any]] = []
        self.max_connections = 1000
        self.max_message_history = 1000
    
    async def add_connection(self, websocket, user_info: Dict[str, Any] = None) -> str:
        """Add a new WebSocket connection"""
        try:
            connection_id = str(uuid.uuid4())
            connection = WebSocketConnection(websocket, connection_id, user_info)
            self.connections[connection_id] = connection
            
            # Send welcome message
            await connection.send_success("Connected to LLaMA Gateway WebSocket", {
                "connection_id": connection_id,
                "features": ["chat", "streaming", "real_time"],
                "server_time": datetime.now().isoformat()
            })
            
            logger.info(f"🔗 WebSocket connection added: {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"❌ Error adding WebSocket connection: {e}")
            return None
    
    async def remove_connection(self, connection_id: str) -> bool:
        """Remove a WebSocket connection"""
        try:
            if connection_id in self.connections:
                # Remove from all rooms
                for room_id, room_connections in self.rooms.items():
                    room_connections.discard(connection_id)
                
                del self.connections[connection_id]
                logger.info(f"🔗 WebSocket connection removed: {connection_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error removing WebSocket connection: {e}")
            return False
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a specific connection"""
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                return await connection.send_message(message)
            return False
            
        except Exception as e:
            logger.error(f"❌ Error sending to connection {connection_id}: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_connections: List[str] = None) -> int:
        """Broadcast a message to all connections"""
        try:
            exclude_connections = exclude_connections or []
            sent_count = 0
            
            for connection_id, connection in self.connections.items():
                if connection_id not in exclude_connections:
                    if await connection.send_message(message):
                        sent_count += 1
            
            logger.info(f"📢 Broadcast message sent to {sent_count} connections")
            return sent_count
            
        except Exception as e:
            logger.error(f"❌ Error broadcasting message: {e}")
            return 0
    
    async def send_to_room(self, room_id: str, message: Dict[str, Any]) -> int:
        """Send a message to all connections in a room"""
        try:
            if room_id not in self.rooms:
                return 0
            
            sent_count = 0
            for connection_id in self.rooms[room_id]:
                if connection_id in self.connections:
                    if await self.send_to_connection(connection_id, message):
                        sent_count += 1
            
            logger.info(f"📢 Room message sent to {sent_count} connections in room {room_id}")
            return sent_count
            
        except Exception as e:
            logger.error(f"❌ Error sending to room {room_id}: {e}")
            return 0
    
    async def join_room(self, connection_id: str, room_id: str) -> bool:
        """Add a connection to a room"""
        try:
            if connection_id not in self.connections:
                return False
            
            if room_id not in self.rooms:
                self.rooms[room_id] = set()
            
            self.rooms[room_id].add(connection_id)
            
            # Notify the connection
            await self.send_to_connection(connection_id, {
                "type": "room_joined",
                "room_id": room_id,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"🔗 Connection {connection_id} joined room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error joining room: {e}")
            return False
    
    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        """Remove a connection from a room"""
        try:
            if room_id in self.rooms and connection_id in self.rooms[room_id]:
                self.rooms[room_id].discard(connection_id)
                
                # Notify the connection
                await self.send_to_connection(connection_id, {
                    "type": "room_left",
                    "room_id": room_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"🔗 Connection {connection_id} left room {room_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error leaving room: {e}")
            return False
    
    async def handle_chat_message(self, connection_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a chat message from a WebSocket connection"""
        try:
            if connection_id not in self.connections:
                return {"error": "Connection not found"}
            
            connection = self.connections[connection_id]
            
            # Extract message content
            user_message = message_data.get("message", "")
            specialist = message_data.get("specialist", "default")
            stream = message_data.get("stream", False)
            
            if not user_message:
                return {"error": "No message content provided"}
            
            # Record message in history
            message_record = {
                "connection_id": connection_id,
                "user_message": user_message,
                "specialist": specialist,
                "timestamp": datetime.now().isoformat(),
                "stream": stream
            }
            self.message_history.append(message_record)
            
            # Keep only last N messages
            if len(self.message_history) > self.max_message_history:
                self.message_history = self.message_history[-self.max_message_history:]
            
            # Generate response (simplified - in real implementation, call LLaMA service)
            if stream:
                # Stream response
                response_tokens = [
                    f"Hello! I'm {specialist} and I received your message: '{user_message}'",
                    "I'm here to help you with your coding and development needs.",
                    "How can I assist you today?"
                ]
                
                for i, token in enumerate(response_tokens):
                    await self.send_to_connection(connection_id, {
                        "type": "stream_token",
                        "token": token,
                        "index": i,
                        "finished": False,
                        "timestamp": datetime.now().isoformat()
                    })
                    await asyncio.sleep(0.1)  # Simulate streaming delay
                
                # Send completion
                await self.send_to_connection(connection_id, {
                    "type": "stream_complete",
                    "timestamp": datetime.now().isoformat()
                })
                
                return {"success": True, "streamed": True}
            else:
                # Regular response
                response = f"Hello! I'm {specialist} and I received your message: '{user_message}'. I'm here to help you with your coding and development needs. How can I assist you today?"
                
                await self.send_to_connection(connection_id, {
                    "type": "chat_response",
                    "response": response,
                    "specialist": specialist,
                    "timestamp": datetime.now().isoformat()
                })
                
                return {"success": True, "response": response}
            
        except Exception as e:
            logger.error(f"❌ Error handling chat message: {e}")
            return {"error": str(e)}
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        try:
            active_connections = len(self.connections)
            total_rooms = len(self.rooms)
            total_messages = len(self.message_history)
            
            # Calculate average messages per connection
            avg_messages = 0
            if active_connections > 0:
                total_connection_messages = sum(conn.message_count for conn in self.connections.values())
                avg_messages = total_connection_messages / active_connections
            
            return {
                "active_connections": active_connections,
                "total_rooms": total_rooms,
                "total_messages": total_messages,
                "average_messages_per_connection": round(avg_messages, 2),
                "max_connections": self.max_connections,
                "max_message_history": self.max_message_history
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting connection stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_inactive_connections(self, timeout_minutes: int = 30) -> int:
        """Clean up inactive connections"""
        try:
            current_time = datetime.now()
            timeout = timedelta(minutes=timeout_minutes)
            cleaned_count = 0
            
            inactive_connections = []
            for connection_id, connection in self.connections.items():
                if current_time - connection.last_activity > timeout:
                    inactive_connections.append(connection_id)
            
            for connection_id in inactive_connections:
                await self.remove_connection(connection_id)
                cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} inactive connections")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up connections: {e}")
            return 0

# Global WebSocket service
websocket_service = WebSocketService()

# Convenience functions
async def add_connection(websocket, user_info: Dict[str, Any] = None) -> str:
    """Add a WebSocket connection"""
    return await websocket_service.add_connection(websocket, user_info)

async def remove_connection(connection_id: str) -> bool:
    """Remove a WebSocket connection"""
    return await websocket_service.remove_connection(connection_id)

async def send_to_connection(connection_id: str, message: Dict[str, Any]) -> bool:
    """Send message to connection"""
    return await websocket_service.send_to_connection(connection_id, message)

async def broadcast_message(message: Dict[str, Any], exclude_connections: List[str] = None) -> int:
    """Broadcast message to all connections"""
    return await websocket_service.broadcast_message(message, exclude_connections)

async def handle_chat_message(connection_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle chat message"""
    return await websocket_service.handle_chat_message(connection_id, message_data)

async def get_connection_stats() -> Dict[str, Any]:
    """Get connection statistics"""
    return await websocket_service.get_connection_stats()
