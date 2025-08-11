#!/usr/bin/env python3
"""
Autonomous MCP Client Agent with Feedback Loop and Memory
Connects to multiple MCP servers and provides intelligent task execution
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MCPServer:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    server_used: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class FeedbackEntry:
    """Feedback entry for learning"""
    task_id: str
    original_approach: str
    feedback: str
    improvement: str
    success_rate: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MemoryManager:
    """Manages persistent memory for the agent"""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Task history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                task_id TEXT PRIMARY KEY,
                task_description TEXT,
                success BOOLEAN,
                result TEXT,
                error TEXT,
                execution_time REAL,
                server_used TEXT,
                timestamp TEXT
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                original_approach TEXT,
                feedback TEXT,
                improvement TEXT,
                success_rate REAL,
                timestamp TEXT
            )
        """)
        
        # Server performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS server_performance (
                server_name TEXT,
                task_type TEXT,
                success_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                avg_execution_time REAL DEFAULT 0.0,
                last_updated TEXT,
                PRIMARY KEY (server_name, task_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_task_result(self, task_result: TaskResult, task_description: str):
        """Store task execution result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO task_history 
            (task_id, task_description, success, result, error, execution_time, server_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_result.task_id,
            task_description,
            task_result.success,
            json.dumps(task_result.result) if task_result.result else None,
            task_result.error,
            task_result.execution_time,
            task_result.server_used,
            task_result.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: FeedbackEntry):
        """Store feedback for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback 
            (task_id, original_approach, feedback, improvement, success_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            feedback.task_id,
            feedback.original_approach,
            feedback.feedback,
            feedback.improvement,
            feedback.success_rate,
            feedback.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def update_server_performance(self, server_name: str, task_type: str, 
                                success: bool, execution_time: float):
        """Update server performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO server_performance 
            (server_name, task_type, success_count, total_count, avg_execution_time, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(server_name, task_type) DO UPDATE SET
                success_count = success_count + ?,
                total_count = total_count + 1,
                avg_execution_time = (avg_execution_time * total_count + ?) / (total_count + 1),
                last_updated = ?
        """, (
            server_name, task_type, 
            1 if success else 0, 1, execution_time, 
            datetime.now().isoformat(),
            1 if success else 0, execution_time,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_server_performance(self, server_name: str = None, task_type: str = None) -> List[Dict]:
        """Get server performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM server_performance"
        params = []
        
        if server_name and task_type:
            query += " WHERE server_name = ? AND task_type = ?"
            params = [server_name, task_type]
        elif server_name:
            query += " WHERE server_name = ?"
            params = [server_name]
        elif task_type:
            query += " WHERE task_type = ?"
            params = [task_type]
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        columns = ['server_name', 'task_type', 'success_count', 
                  'total_count', 'avg_execution_time', 'last_updated']
        return [dict(zip(columns, row)) for row in results]
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get recent feedback entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM feedback 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'task_id', 'original_approach', 'feedback', 
                  'improvement', 'success_rate', 'timestamp']
        return [dict(zip(columns, row)) for row in results]

class MCPClientManager:
    """Manages connections to multiple MCP servers"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.clients: Dict[str, ClientSession] = {}
        self.server_capabilities: Dict[str, Dict] = {}
    
    def add_server(self, server: MCPServer):
        """Add an MCP server configuration"""
        self.servers[server.name] = server
        logger.info(f"Added MCP server: {server.name}")
    
    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        for server_name, server_config in self.servers.items():
            try:
                await self._connect_to_server(server_name, server_config)
            except Exception as e:
                logger.error(f"Failed to connect to server {server_name}: {e}")
    
    async def _connect_to_server(self, server_name: str, server_config: MCPServer):
        """Connect to a single MCP server"""
        server_params = StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            env=server_config.env
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Store the session
                self.clients[server_name] = session
                
                # Get server capabilities
                try:
                    tools = await session.list_tools()
                    resources = await session.list_resources()
                    
                    self.server_capabilities[server_name] = {
                        'tools': [tool.dict() for tool in tools],
                        'resources': [resource.dict() for resource in resources],
                        'connected': True
                    }
                    
                    logger.info(f"Connected to {server_name}: {len(tools)} tools, {len(resources)} resources")
                except Exception as e:
                    logger.warning(f"Could not get capabilities for {server_name}: {e}")
                    self.server_capabilities[server_name] = {'connected': True, 'error': str(e)}
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool on a specific server"""
        if server_name not in self.clients:
            raise Exception(f"Server {server_name} not connected")
        
        session = self.clients[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result
    
    def get_available_tools(self, server_name: str = None) -> Dict:
        """Get available tools from servers"""
        if server_name:
            return self.server_capabilities.get(server_name, {}).get('tools', [])
        
        all_tools = {}
        for name, capabilities in self.server_capabilities.items():
            all_tools[name] = capabilities.get('tools', [])
        return all_tools
    
    async def disconnect_all(self):
        """Disconnect from all servers"""
        for server_name in list(self.clients.keys()):
            try:
                # Note: In a real implementation, you'd properly close the sessions
                del self.clients[server_name]
                logger.info(f"Disconnected from {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")

class FeedbackLoop:
    """Implements feedback loop for continuous learning"""
    
    def __init__(self, memory_manager: MemoryManager, openai_client):
        self.memory = memory_manager
        self.openai_client = openai_client
    
    async def analyze_task_result(self, task_result: TaskResult, 
                                task_description: str, 
                                original_approach: str) -> FeedbackEntry:
        """Analyze task result and generate feedback"""
        
        # Get recent similar tasks for context
        recent_feedback = self.memory.get_recent_feedback(5)
        
        # Create prompt for AI analysis
        prompt = f"""
        Analyze the following task execution and provide feedback:
        
        Task: {task_description}
        Original Approach: {original_approach}
        Success: {task_result.success}
        Result: {task_result.result}
        Error: {task_result.error}
        Execution Time: {task_result.execution_time}s
        Server Used: {task_result.server_used}
        
        Recent Feedback Context:
        {json.dumps(recent_feedback, indent=2)}
        
        Please provide:
        1. Feedback on what went well or what went wrong
        2. Specific improvements for future similar tasks
        3. A success rate prediction (0.0-1.0) for this approach
        
        Respond in JSON format:
        {{
            "feedback": "detailed feedback",
            "improvement": "specific improvement suggestions",
            "success_rate": 0.0-1.0
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            feedback_entry = FeedbackEntry(
                task_id=task_result.task_id,
                original_approach=original_approach,
                feedback=analysis['feedback'],
                improvement=analysis['improvement'],
                success_rate=analysis['success_rate']
            )
            
            self.memory.store_feedback(feedback_entry)
            return feedback_entry
            
        except Exception as e:
            logger.error(f"Error analyzing task result: {e}")
            # Return default feedback
            return FeedbackEntry(
                task_id=task_result.task_id,
                original_approach=original_approach,
                feedback="Analysis failed",
                improvement="Ensure proper error handling",
                success_rate=0.5 if task_result.success else 0.1
            )

class AutonomousMCPAgent:
    """Main autonomous MCP client agent"""
    
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.memory = MemoryManager()
        self.mcp_manager = MCPClientManager()
        self.feedback_loop = FeedbackLoop(self.memory, self.openai_client)
        self.running = False
        
    def add_mcp_server(self, server: MCPServer):
        """Add an MCP server to the agent"""
        self.mcp_manager.add_server(server)
    
    async def start(self):
        """Start the autonomous agent"""
        logger.info("Starting Autonomous MCP Agent...")
        
        # Connect to all MCP servers
        await self.mcp_manager.connect_to_servers()
        
        self.running = True
        logger.info("Agent started successfully!")
    
    async def stop(self):
        """Stop the autonomous agent"""
        logger.info("Stopping Autonomous MCP Agent...")
        self.running = False
        await self.mcp_manager.disconnect_all()
        logger.info("Agent stopped.")
    
    async def execute_task(self, task_description: str, 
                          preferred_server: str = None) -> TaskResult:
        """Execute a task autonomously"""
        task_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze the task and determine the best approach
            approach = await self._plan_task_execution(task_description, preferred_server)
            
            # Execute the planned approach
            result = await self._execute_planned_approach(approach)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            task_result = TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                server_used=approach.get('server_name')
            )
            
            # Store result and generate feedback
            self.memory.store_task_result(task_result, task_description)
            await self.feedback_loop.analyze_task_result(
                task_result, task_description, str(approach)
            )
            
            # Update server performance
            if approach.get('server_name'):
                self.memory.update_server_performance(
                    approach['server_name'], 
                    approach.get('task_type', 'general'),
                    True, 
                    execution_time
                )
            
            return task_result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            task_result = TaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
            
            self.memory.store_task_result(task_result, task_description)
            logger.error(f"Task execution failed: {e}")
            
            return task_result
    
    async def _plan_task_execution(self, task_description: str, 
                                 preferred_server: str = None) -> Dict:
        """Plan how to execute a task"""
        
        # Get available tools and server performance
        available_tools = self.mcp_manager.get_available_tools()
        server_performance = self.memory.get_server_performance()
        recent_feedback = self.memory.get_recent_feedback(5)
        
        # Create planning prompt
        prompt = f"""
        Plan how to execute the following task using available MCP servers and tools:
        
        Task: {task_description}
        Preferred Server: {preferred_server or 'None'}
        
        Available Tools by Server:
        {json.dumps(available_tools, indent=2)}
        
        Server Performance History:
        {json.dumps(server_performance, indent=2)}
        
        Recent Feedback:
        {json.dumps(recent_feedback, indent=2)}
        
        Please provide an execution plan in JSON format:
        {{
            "server_name": "best server to use",
            "tool_name": "tool to execute",
            "arguments": {{"arg1": "value1"}},
            "task_type": "category of this task",
            "reasoning": "why this approach was chosen"
        }}
        """
        
        try:
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            # Validate the plan
            if plan['server_name'] not in self.mcp_manager.clients:
                raise Exception(f"Server {plan['server_name']} not available")
            
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Return a default plan
            if available_tools:
                first_server = next(iter(available_tools.keys()))
                first_tool = available_tools[first_server][0] if available_tools[first_server] else None
                
                return {
                    'server_name': first_server,
                    'tool_name': first_tool['name'] if first_tool else 'unknown',
                    'arguments': {},
                    'task_type': 'general',
                    'reasoning': 'Fallback plan due to planning error'
                }
            else:
                raise Exception("No available tools for task execution")
    
    async def _execute_planned_approach(self, approach: Dict) -> Any:
        """Execute the planned approach"""
        server_name = approach['server_name']
        tool_name = approach['tool_name']
        arguments = approach['arguments']
        
        logger.info(f"Executing {tool_name} on {server_name} with args: {arguments}")
        
        result = await self.mcp_manager.execute_tool(server_name, tool_name, arguments)
        return result
    
    async def get_status(self) -> Dict:
        """Get agent status"""
        return {
            'running': self.running,
            'connected_servers': list(self.mcp_manager.clients.keys()),
            'server_capabilities': self.mcp_manager.server_capabilities,
            'recent_feedback': self.memory.get_recent_feedback(3),
            'server_performance': self.memory.get_server_performance()
        }

# Example usage and configuration
async def main():
    """Example usage of the Autonomous MCP Agent"""
    
    # Initialize the agent with OpenAI API key
    agent = AutonomousMCPAgent(openai_api_key=os.getenv('OPENAI_API_KEY'))
    # agent = MCPAgent(
    #     openai_api_key=os.getenv('OPENAI_API_KEY'),
    #     server_configs=server_configs
    # )
    # Configure MCP servers
    filesystem_server = MCPServer(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"],
        description="File system operations"
    )
    
    sqlite_server = MCPServer(
        name="sqlite", 
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "./example.db"],
        description="SQLite database operations"
    )
    
    web_server = MCPServer(
        name="web",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-web"],
        description="Web scraping and HTTP requests"
    )
    
    # Add servers to agent
    agent.add_mcp_server(filesystem_server)
    agent.add_mcp_server(sqlite_server) 
    agent.add_mcp_server(web_server)
    
    try:
        # Start the agent
        await agent.start()
        
        # Execute some tasks
        tasks = [
            "Create a new SQLite table called 'users' with columns id, name, email",
            "Fetch the content from https://httpbin.org/json",
            "Search for Python files and count their lines of code"
        ]
        
        for task in tasks:
            print(f"\n--- Executing Task: {task} ---")
            result = await agent.execute_task(task)
            print(f"Success: {result.success}")
            print(f"Result: {result.result}")
            if result.error:
                print(f"Error: {result.error}")
            print(f"Execution time: {result.execution_time:.2f}s")
            
            # Wait a bit between tasks
            await asyncio.sleep(1)
        
        # Print agent status
        status = await agent.get_status()
        print(f"\n--- Agent Status ---")
        print(json.dumps(status, indent=2, default=str))
        
    finally:
        # Stop the agent
        await agent.stop()

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())