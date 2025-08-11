import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    description: str
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class MCPServer:
    name: str
    command: List[str]
    args: List[str]
    env: Optional[Dict[str, str]] = None
    session: Optional[ClientSession] = None
    available_tools: List[Dict[str, Any]] = None

class AutonomousMCPClient:
    def __init__(self, openai_api_key: str):
        """Initialize the autonomous MCP client with OpenAI API key."""
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.servers: Dict[str, MCPServer] = {}
        self.tasks: List[Task] = []
        self.task_counter = 0
        
    async def add_server(self, server: MCPServer) -> bool:
        """Add and connect to an MCP server."""
        try:
            logger.info(f"Connecting to server: {server.name}")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=server.command[0],
                args=server.command[1:] + server.args,
                env=server.env or {}
            )
            
            # Connect to the server
            stdio_transport = await stdio_client(server_params)
            session = ClientSession(stdio_transport[0], stdio_transport[1])
            
            # Initialize the session
            await session.initialize()
            
            # Get available tools
            tools_result = await session.list_tools()
            server.available_tools = tools_result.tools if hasattr(tools_result, 'tools') else []
            server.session = session
            
            self.servers[server.name] = server
            logger.info(f"Connected to {server.name} with {len(server.available_tools)} tools")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to server {server.name}: {str(e)}")
            return False
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server."""
        if server_name in self.servers:
            server = self.servers[server_name]
            if server.session:
                try:
                    # Close the session gracefully
                    await server.session.close()
                except Exception as e:
                    logger.warning(f"Error closing session for {server_name}: {str(e)}")
            
            del self.servers[server_name]
            logger.info(f"Disconnected from server: {server_name}")
    
    def add_task(self, description: str, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Add a new task to the queue."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        task = Task(
            id=task_id,
            description=description,
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments
        )
        
        self.tasks.append(task)
        logger.info(f"Added task {task_id}: {description}")
        return task_id
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on a specific MCP server."""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not found")
        
        server = self.servers[server_name]
        if not server.session:
            raise ValueError(f"Server {server_name} not connected")
        
        # Verify tool exists
        tool_exists = any(tool.name == tool_name for tool in server.available_tools)
        if not tool_exists:
            available_tools = [tool.name for tool in server.available_tools]
            raise ValueError(f"Tool {tool_name} not found on server {server_name}. Available tools: {available_tools}")
        
        try:
            # Execute the tool
            result = await server.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} on {server_name}: {str(e)}")
            raise
    
    async def analyze_task_with_ai(self, task: Task) -> Dict[str, Any]:
        """Use OpenAI to analyze and potentially modify task parameters."""
        try:
            server = self.servers.get(task.server_name)
            if not server:
                return {"success": False, "error": "Server not found"}
            
            # Get tool schema
            tool_schema = next((tool for tool in server.available_tools if tool.name == task.tool_name), None)
            if not tool_schema:
                return {"success": False, "error": "Tool not found"}
            
            # Create a prompt for the AI to analyze the task
            prompt = f"""
            Analyze this task and verify the arguments are correct:
            
            Task: {task.description}
            Tool: {task.tool_name}
            Server: {task.server_name}
            Arguments: {json.dumps(task.arguments, indent=2)}
            
            Tool Schema: {json.dumps(tool_schema.inputSchema if hasattr(tool_schema, 'inputSchema') else {}, indent=2)}
            
            Please respond with a JSON object containing:
            - "valid": boolean indicating if arguments are valid
            - "suggestions": any improvements to the arguments
            - "modified_arguments": corrected arguments if needed
            - "reasoning": explanation of any changes
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing MCP tool calls. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            return {"success": True, "analysis": ai_response}
            
        except Exception as e:
            logger.error(f"AI analysis failed for task {task.id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def execute_task(self, task: Task) -> bool:
        """Execute a single task."""
        try:
            task.status = TaskStatus.IN_PROGRESS
            logger.info(f"Executing task {task.id}: {task.description}")
            
            # Analyze task with AI first
            ai_analysis = await self.analyze_task_with_ai(task)
            if ai_analysis["success"] and ai_analysis["analysis"].get("modified_arguments"):
                logger.info(f"AI suggested argument modifications for task {task.id}")
                task.arguments = ai_analysis["analysis"]["modified_arguments"]
            
            # Execute the tool
            result = await self.execute_tool(task.server_name, task.tool_name, task.arguments)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            logger.info(f"Task {task.id} completed successfully")
            return True
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            logger.error(f"Task {task.id} failed: {str(e)}")
            return False
    
    async def execute_all_tasks(self, max_concurrent: int = 3) -> Dict[str, Any]:
        """Execute all pending tasks with concurrency control."""
        pending_tasks = [task for task in self.tasks if task.status == TaskStatus.PENDING]
        
        if not pending_tasks:
            logger.info("No pending tasks to execute")
            return {"completed": 0, "failed": 0, "total": 0}
        
        logger.info(f"Executing {len(pending_tasks)} tasks with max concurrency of {max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task):
            async with semaphore:
                return await self.execute_task(task)
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in pending_tasks],
            return_exceptions=True
        )
        
        # Count results
        completed = sum(1 for result in results if result is True)
        failed = len(results) - completed
        
        logger.info(f"Task execution completed: {completed} successful, {failed} failed")
        
        return {
            "completed": completed,
            "failed": failed,
            "total": len(pending_tasks)
        }
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for task in self.tasks if task.status == status)
        
        return {
            "task_counts": status_counts,
            "total_tasks": len(self.tasks),
            "tasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "server": task.server_name,
                    "tool": task.tool_name,
                    "status": task.status.value,
                    "result": task.result if task.status == TaskStatus.COMPLETED else None,
                    "error": task.error if task.status == TaskStatus.FAILED else None
                }
                for task in self.tasks
            ]
        }
    
    async def close(self):
        """Close all server connections."""
        for server_name in list(self.servers.keys()):
            await self.disconnect_server(server_name)

# Example usage and configuration
async def main():
    """Example usage of the Autonomous MCP Client."""
    
    # Initialize the client with OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    client = AutonomousMCPClient(openai_api_key)
    
    try:
        # Example: Add file system server
        file_server = MCPServer(
            name="filesystem",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            args=["/tmp"]  # Root directory for file operations
        )
        
        # Example: Add GitHub server (if available)
        github_server = MCPServer(
            name="github",
            command=["npx", "-y", "@modelcontextprotocol/server-github"],
            args=[],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")}
        )
        
        # Connect to servers
        await client.add_server(file_server)
        # await client.add_server(github_server)  # Uncomment if you have GitHub token
        
        # Add some example tasks
        client.add_task(
            description="List files in the root directory",
            server_name="filesystem",
            tool_name="read_file",  # This would need to match actual tool name
            arguments={"path": "/tmp"}
        )
        
        client.add_task(
            description="Create a test file",
            server_name="filesystem", 
            tool_name="write_file",  # This would need to match actual tool name
            arguments={"path": "/tmp/test.txt", "content": "Hello from MCP client!"}
        )
        
        # Execute all tasks
        results = await client.execute_all_tasks(max_concurrent=2)
        
        # Print results
        print("\n=== Execution Results ===")
        print(f"Total tasks: {results['total']}")
        print(f"Completed: {results['completed']}")
        print(f"Failed: {results['failed']}")
        
        # Print detailed task status
        status = client.get_task_status()
        print("\n=== Task Details ===")
        for task in status["tasks"]:
            print(f"Task {task['id']}: {task['status']}")
            print(f"  Description: {task['description']}")
            if task['result']:
                print(f"  Result: {task['result']}")
            if task['error']:
                print(f"  Error: {task['error']}")
            print()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    
    finally:
        # Clean up
        await client.close()

if __name__ == "__main__":
    # Install required packages
    required_packages = [
        "openai",
        "mcp",
        "asyncio-mqtt"  # If needed for other protocols
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
    
    # Run the main function
    asyncio.run(main())