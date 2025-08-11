import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import AsyncExitStack
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMemory:
    """Memory structure for the agent"""
    task_history: List[Dict[str, Any]]
    learned_patterns: Dict[str, Any]
    successful_tool_combinations: List[List[str]]
    failed_attempts: List[Dict[str, Any]]
    resource_cache: Dict[str, Any]
    
    def add_task(self, task: str, result: Any, tools_used: List[str], success: bool):
        """Add a completed task to memory"""
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'result': result,
            'tools_used': tools_used,
            'success': success
        })
        
        if success and tools_used:
            self.successful_tool_combinations.append(tools_used)
        elif not success:
            self.failed_attempts.append({
                'task': task,
                'tools_used': tools_used,
                'timestamp': datetime.now().isoformat()
            })

class MCPAgent:
    """Autonomous MCP Client Agent with feedback loop and memory"""
    
    def __init__(self, openai_api_key: str, server_configs: List[Dict[str, Any]]):
        """
        Initialize the MCP Agent
        
        Args:
            openai_api_key: OpenAI API key
            server_configs: List of MCP server configurations
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.server_configs = server_configs
        self.sessions = {}
        self.memory = AgentMemory(
            task_history=[],
            learned_patterns={},
            successful_tool_combinations=[],
            failed_attempts=[],
            resource_cache={}
        )
        self.available_tools = {}
        self.available_tools_list = []
        self.available_prompts = {}
        self.available_resources = {}
        
    async def initialize_servers(self):
        """Initialize connections to all MCP servers"""
        for config in self.server_configs:
            print(config)
            try:
                server_params = StdioServerParameters(
                    command=config['command'],
                    args=config.get('args', []),
                    env=config.get('env', None)
                )
                print("server_params", server_params)
                # session = await stdio_client(server_params)
                # await session.initialize()

                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                self.stdio, self.write = stdio_transport
                async_session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
                
                await async_session.initialize()
                
                # List available tools
                response = await async_session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.sessions[config['name']] = async_session
                logger.info(f"Connected to MCP server: {config['name']}")
                
                # Discover capabilities
                await self._discover_server_capabilities(config['name'], async_session)
                
            except Exception as e:
                logger.error(f"Failed to connect to server {config['name']}: {e}")
    
    async def _discover_server_capabilities(self, server_name: str, session: ClientSession):
        """Discover tools, prompts, and resources from an MCP server"""
        try:
            # Get available tools
            tools_result = await session.list_tools()
            if tools_result.tools:
                self.available_tools[server_name] = {
                    tool.name: {
                        'description': tool.description,
                        'input_schema': tool.inputSchema
                    } for tool in tools_result.tools
                }

            # Get available tools list
            if tools_result.tools:
                for tool in tools_result.tools:
                    self.available_tools_list.append({ 
                                "type": "function",  # OpenAI requires "type": "function"
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.inputSchema  # OpenAI uses "parameters" instead of "input_schema"
                                }
                             })
                logger.info(f"Discovered {len(tools_result.tools)} tools from {server_name}")
            
            # Get available prompts
            prompts_result = await session.list_prompts()
            if prompts_result.prompts:
                self.available_prompts[server_name] = {
                    prompt.name: {
                        'description': prompt.description,
                        'arguments': prompt.arguments
                    } for prompt in prompts_result.prompts
                }
                logger.info(f"Discovered {len(prompts_result.prompts)} prompts from {server_name}")
            
            # Get available resources
            resources_result = await session.list_resources()
            if resources_result.resources:
                self.available_resources[server_name] = {
                    resource.uri: {
                        'name': resource.name,
                        'description': resource.description,
                        'mimeType': resource.mimeType
                    } for resource in resources_result.resources
                }
                logger.info(f"Discovered {len(resources_result.resources)} resources from {server_name}")
                
        except Exception as e:
            logger.error(f"Error discovering capabilities for {server_name}: {e}")
    
    async def _analyze_task_with_llm(self, task: str) -> Dict[str, Any]:
        """Use LLM to analyze the task and determine the best approach"""
        # Create context from available capabilities
        capabilities_context = self._build_capabilities_context()
        memory_context = self._build_memory_context()
        
        analysis_prompt = f"""
        You are an intelligent agent that needs to complete the following task using available MCP tools, prompts, and resources.
        
        TASK: {task}
        
        AVAILABLE CAPABILITIES:
        {capabilities_context}
        
        MEMORY FROM PAST TASKS:
        {memory_context}
        
        Based on this information, provide a JSON response with:
        1. "strategy": A step-by-step plan to complete the task
        2. "tools_needed": List of tools that should be used
        3. "resources_needed": List of resources that should be accessed
        4. "prompts_needed": List of prompts that should be used
        5. "confidence": Your confidence level (0-1) in this approach
        6. "reasoning": Explanation of your approach
        
        Return ONLY valid JSON.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3,
        )
        
        try:
            print(response)
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON")
            return {"strategy": ["fallback"], "tools_needed": [], "resources_needed": [], "prompts_needed": [], "confidence": 0.1, "reasoning": "Fallback due to parsing error"}
    
    def _build_capabilities_context(self) -> str:
        """Build context string of available capabilities"""
        context = "TOOLS:\n"
        for server, tools in self.available_tools.items():
            for tool_name, tool_info in tools.items():
                context += f"- {server}.{tool_name}: {tool_info['description']}\n"

        # for tool in self.available_tools_list:
        #     context += str(tool)
        
        context += "\nPROMPTS:\n"
        for server, prompts in self.available_prompts.items():
            for prompt_name, prompt_info in prompts.items():
                context += f"- {server}.{prompt_name}: {prompt_info['description']}\n"
        
        context += "\nRESOURCES:\n"
        for server, resources in self.available_resources.items():
            for uri, resource_info in resources.items():
                context += f"- {server}: {resource_info['name']} - {resource_info['description']}\n"
        
        return context
    
    def _build_memory_context(self) -> str:
        """Build context from agent's memory"""
        context = f"SUCCESSFUL TOOL COMBINATIONS: {self.memory.successful_tool_combinations[-5:]}\n"
        context += f"RECENT TASKS: {[t['task'] for t in self.memory.task_history[-3:]]}\n"
        return context
    
    async def _execute_strategy(self, strategy: Dict[str, Any]) -> Tuple[Any, bool]:
        """Execute the determined strategy"""
        results = []
        tools_used = []

        try:
            # Execute each step in the strategy
            for step in strategy['strategy']:
                logger.info(f"Executing step: {step}")
                


            # Use needed tools
            for tool_spec in strategy['tools_needed']:
                if '.' in tool_spec:
                    server_name, tool_name = tool_spec.split('.', 1)
                    result = await self._use_tool(server_name, tool_name, {})
                    results.append(result)
                    tools_used.append(tool_spec)

            # Access needed resources
            for resource_spec in strategy['resources_needed']:
                if '.' in resource_spec:
                    server_name, resource_uri = resource_spec.split('.', 1)
                    result = await self._access_resource(server_name, resource_uri)
                    results.append(result)
            
            # Use needed prompts
            for prompt_spec in strategy['prompts_needed']:
                if '.' in prompt_spec:
                    server_name, prompt_name = prompt_spec.split('.', 1)
                    result = await self._use_prompt(server_name, prompt_name, {})
                    results.append(result)
                    tools_used.append(prompt_spec)
            
            return results, True
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return str(e), False
    
    async def _use_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any], task: str) -> Any:
        """Use a specific tool from an MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result.content if result.content else "Tool executed successfully"
    
    async def _access_resource(self, server_name: str, resource_uri: str) -> Any:
        """Access a specific resource from an MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.sessions[server_name]
        result = await session.read_resource(resource_uri)
        return result.contents if result.contents else "Resource accessed successfully"
    
    async def _use_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> Any:
        """Use a specific prompt from an MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.sessions[server_name]
        result = await session.get_prompt(prompt_name, arguments)
        return result.messages if result.messages else "Prompt executed successfully"
    
    async def _evaluate_results_with_llm(self, task: str, results: Any, success: bool) -> Dict[str, Any]:
        """Use LLM to evaluate the results and provide feedback"""
        evaluation_prompt = f"""
        Evaluate the results of this task execution:
        
        ORIGINAL TASK: {task}
        EXECUTION SUCCESS: {success}
        RESULTS: {str(results)[:1000]}...
        
        Provide a JSON response with:
        1. "task_completed": boolean - was the original task successfully completed?
        2. "quality_score": float (0-1) - quality of the results
        3. "lessons_learned": list of insights for future similar tasks
        4. "improvements": list of suggestions for better execution
        5. "success_factors": list of what worked well
        
        Return ONLY valid JSON.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.3
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"task_completed": success, "quality_score": 0.5, "lessons_learned": [], "improvements": [], "success_factors": []}
    
    async def execute_task(self, task: str) -> Dict[str, Any]:
        """Main method to execute a task autonomously"""
        logger.info(f"Starting task execution: {task}")
        
        try:
            # Step 1: Analyze task with LLM
            analysis = await self._analyze_task_with_llm(task)
            logger.info(f"Task analysis completed with confidence: {analysis.get('confidence', 0)}")
            
            # Step 2: Execute the determined strategy
            results, execution_success = await self._execute_strategy(analysis)
            
            # Step 3: Evaluate results with LLM
            evaluation = await self._evaluate_results_with_llm(task, results, execution_success)
            
            # Step 4: Update memory
            tools_used = analysis.get('tools_needed', []) + analysis.get('prompts_needed', [])
            final_success = evaluation.get('task_completed', False)
            self.memory.add_task(task, results, tools_used, final_success)
            
            # Step 5: Learn from the experience
            if evaluation.get('lessons_learned'):
                for lesson in evaluation['lessons_learned']:
                    self.memory.learned_patterns[task[:50]] = lesson
            
            return {
                'task': task,
                'success': final_success,
                'results': results,
                'analysis': analysis,
                'evaluation': evaluation,
                'tools_used': tools_used
            }
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            self.memory.add_task(task, str(e), [], False)
            return {
                'task': task,
                'success': False,
                'results': str(e),
                'error': True
            }
    
    async def close(self):
        """Close all MCP server connections"""
        for session in self.sessions.values():
            await session.close()

# Example usage and configuration
async def main():
    """Example of how to use the MCP Agent"""
    
    # Configuration for MCP servers
    server_configs = [
    {
        'name': 'weather',
        'command': 'python',
        'args': ["C:\\Anuj\\AI\\MCP\\mcp-servers\\weather\\server.py"]
    },
    {
        'name': 'duckdb',
        'command': 'python',
        'args': ["C:\\Anuj\\AI\\MCP\\mcp-servers\\duckdb\\server.py"]
    },
    {
        'name': 'financial',
        'command': 'python',
        'args': ["C:\\Anuj\\AI\\MCP\\mcp-servers\\financial_datasets\\server.py"]
    }
    ]
    
    # Initialize the agent
    agent = MCPAgent(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        server_configs=server_configs
    )
    
    try:
        # Initialize MCP servers
        await agent.initialize_servers()
        
        # Execute a task
        task = "What is the weaqther in new your and insert into duckdb table name weather"
        result = await agent.execute_task(task)
        
        print(f"Task: {result['task']}")
        print(f"Success: {result['success']}")
        print(f"Results: {result['results']}")
        
        if 'evaluation' in result:
            print(f"Quality Score: {result['evaluation'].get('quality_score', 'N/A')}")
        
    finally:
        await agent.close()

if __name__ == "__main__":
    # Set your OpenAI API key
    #os.environ['OPENAI_API_KEY'] = 
    
    # Run the agent
    asyncio.run(main())