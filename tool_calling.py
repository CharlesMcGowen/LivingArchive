#!/usr/bin/env python3
"""
Tool Calling Framework for LLaMA Gateway
========================================

Provides function calling capabilities for AI actions.
Enables AI to execute functions, call APIs, and perform actions.

Author: EGO Revolution Team
Version: 1.0.0 - Phase 2 Core Features
"""

import asyncio
import logging
import json
import time
import subprocess
import requests
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

class Tool:
    """Base class for tools that can be called by AI"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function: Optional[Callable] = None
    
    def set_function(self, func: Callable):
        """Set the function to execute"""
        self.function = func
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        try:
            if not self.function:
                return {"error": f"Tool {self.name} has no function set"}
            
            # Validate parameters
            for param_name, param_info in self.parameters.items():
                if param_name in kwargs:
                    # Basic type validation
                    expected_type = param_info.get("type", "string")
                    if expected_type == "string" and not isinstance(kwargs[param_name], str):
                        kwargs[param_name] = str(kwargs[param_name])
                    elif expected_type == "number" and not isinstance(kwargs[param_name], (int, float)):
                        try:
                            kwargs[param_name] = float(kwargs[param_name])
                        except ValueError:
                            return {"error": f"Invalid parameter type for {param_name}"}
            
            # Execute the function
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)
            
            return {
                "tool": self.name,
                "result": result,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Tool execution error for {self.name}: {e}")
            return {
                "tool": self.name,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_builtin_tools()
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"🔧 Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def _register_builtin_tools(self):
        """Register built-in tools"""
        
        # Calculator tool
        calculator = Tool(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "required": True
                }
            }
        )
        calculator.set_function(self._calculator_function)
        self.register_tool(calculator)
        
        # File operations tool
        file_ops = Tool(
            name="file_operations",
            description="Perform file operations (read, write, list)",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (read, write, list)",
                    "required": True
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path",
                    "required": True
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write operation)",
                    "required": False
                }
            }
        )
        file_ops.set_function(self._file_operations_function)
        self.register_tool(file_ops)
        
        # Web search tool
        web_search = Tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": True
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results",
                    "required": False
                }
            }
        )
        web_search.set_function(self._web_search_function)
        self.register_tool(web_search)
        
        # System information tool
        system_info = Tool(
            name="system_info",
            description="Get system information",
            parameters={
                "info_type": {
                    "type": "string",
                    "description": "Type of information to get (cpu, memory, disk, network)",
                    "required": True
                }
            }
        )
        system_info.set_function(self._system_info_function)
        self.register_tool(system_info)
        
        # Email tool (simulated)
        email_tool = Tool(
            name="send_email",
            description="Send an email",
            parameters={
                "to": {
                    "type": "string",
                    "description": "Recipient email address",
                    "required": True
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject",
                    "required": True
                },
                "body": {
                    "type": "string",
                    "description": "Email body",
                    "required": True
                }
            }
        )
        email_tool.set_function(self._send_email_function)
        self.register_tool(email_tool)
    
    async def _calculator_function(self, expression: str) -> str:
        """Calculator function"""
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _file_operations_function(self, operation: str, path: str, content: str = "") -> str:
        """File operations function"""
        try:
            file_path = Path(path)
            
            if operation == "read":
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        return f"File content:\n{f.read()}"
                else:
                    return "Error: File not found"
            
            elif operation == "write":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)
                return f"File written successfully: {path}"
            
            elif operation == "list":
                if file_path.exists() and file_path.is_dir():
                    files = [f.name for f in file_path.iterdir()]
                    return f"Directory contents:\n" + "\n".join(files)
                else:
                    return "Error: Directory not found"
            
            else:
                return "Error: Invalid operation"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _web_search_function(self, query: str, max_results: int = 5) -> str:
        """Web search function (simulated)"""
        try:
            # Simulate web search
            await asyncio.sleep(0.1)
            
            # Mock search results
            results = [
                f"Search result 1 for '{query}': This is a simulated search result that provides relevant information about the topic.",
                f"Search result 2 for '{query}': Another simulated result that contains useful details and insights.",
                f"Search result 3 for '{query}': A third result that offers additional context and information."
            ]
            
            return f"Web search results for '{query}':\n" + "\n\n".join(results[:max_results])
            
        except Exception as e:
            return f"Search error: {str(e)}"
    
    async def _system_info_function(self, info_type: str) -> str:
        """System information function"""
        try:
            if info_type == "cpu":
                return "CPU Information: AMD Ryzen processor with multiple cores"
            elif info_type == "memory":
                return "Memory Information: 32GB RAM available"
            elif info_type == "disk":
                return "Disk Information: 1TB SSD with 500GB free space"
            elif info_type == "network":
                return "Network Information: Connected to internet with good speed"
            else:
                return "Error: Invalid info type"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _send_email_function(self, to: str, subject: str, body: str) -> str:
        """Send email function (simulated)"""
        try:
            # Simulate email sending
            await asyncio.sleep(0.2)
            
            return f"Email sent successfully to {to} with subject '{subject}'"
            
        except Exception as e:
            return f"Email error: {str(e)}"

class ToolCallingService:
    """Service for managing tool calling"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.call_history = []
    
    async def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        try:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                return {"error": f"Tool {tool_name} not found"}
            
            # Execute the tool
            result = await tool.execute(**parameters)
            
            # Record the call
            self.call_history.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool calling error: {e}")
            return {"error": str(e)}
    
    async def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools"""
        return {
            "tools": self.registry.list_tools(),
            "total_tools": len(self.registry.tools),
            "call_history_count": len(self.call_history)
        }
    
    async def get_call_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get tool call history"""
        return self.call_history[-limit:] if self.call_history else []

# Global tool calling service instance
tool_calling_service = ToolCallingService()

# Convenience functions
async def execute_tool_call(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call"""
    return await tool_calling_service.execute_tool_call(tool_name, parameters)

async def get_available_tools() -> Dict[str, Any]:
    """Get available tools"""
    return await tool_calling_service.get_available_tools()

async def get_tool_call_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get tool call history"""
    return await tool_calling_service.get_call_history(limit)
