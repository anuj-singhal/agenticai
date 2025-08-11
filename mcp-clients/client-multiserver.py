from openai import OpenAI
from typing import Optional, Dict, List
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import sys
import asyncio

class MCPClient:
    def __init__(self):
        # Initialize session and client objects for multiple servers
        self.servers: Dict[str, Dict] = {}  # server_name -> {session, stdio, write}
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()

    async def connect_to_server(self, server_script_path: str, server_name: Optional[str] = None):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
            server_name: Optional name for the server (defaults to filename)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        # Generate server name if not provided
        if server_name is None:
            server_name = server_script_path.split('/')[-1].replace('.py', '').replace('.js', '')
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        await session.initialize()
        
        # Store server connection
        self.servers[server_name] = {
            'session': session,
            'stdio': stdio,
            'write': write,
            'path': server_script_path
        }
        
        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])
        
        return server_name

    async def connect_to_multiple_servers(self, server_paths: List[str]):
        """Connect to multiple MCP servers
        
        Args:
            server_paths: List of paths to server scripts
        """
        for path in server_paths:
            try:
                await self.connect_to_server(path)
            except Exception as e:
                print(f"Failed to connect to server {path}: {e}")

    def list_servers(self):
        """List all connected servers and their tools"""
        if not self.servers:
            print("No servers connected.")
            return
            
        print("\nConnected servers:")
        for server_name, server_info in self.servers.items():
            print(f"- {server_name} ({server_info['path']})")

    async def get_all_tools(self):
        """Get all tools from all connected servers"""
        all_tools = []
        
        for server_name, server_info in self.servers.items():
            session = server_info['session']
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    all_tools.append({
                        "type": "function",
                        "function": {
                            "name": f"{server_name}_{tool.name}",  # Prefix with server name
                            "description": f"[{server_name}] {tool.description}",
                            "parameters": tool.inputSchema
                        }
                    })
            except Exception as e:
                print(f"Error getting tools from {server_name}: {e}")
                
        return all_tools

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Call a tool on the appropriate server"""
        # Parse server name from tool name
        if '_' in tool_name:
            server_name, actual_tool_name = tool_name.split('_', 1)
        else:
            # If no server prefix, try to find the tool in any server
            for srv_name, server_info in self.servers.items():
                session = server_info['session']
                try:
                    response = await session.list_tools()
                    tool_names = [tool.name for tool in response.tools]
                    if tool_name in tool_names:
                        server_name = srv_name
                        actual_tool_name = tool_name
                        break
                except:
                    continue
            else:
                raise ValueError(f"Tool {tool_name} not found in any server")
        
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
            
        session = self.servers[server_name]['session']
        return await session.call_tool(actual_tool_name, tool_args)

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools from all servers"""
        if not self.servers:
            return "No servers connected. Please connect to at least one MCP server."
            
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Get all tools from all servers
        available_tools = await self.get_all_tools()

        if not available_tools:
            return "No tools available from connected servers."

        # Initial OpenAI API call
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        message = response.choices[0].message
        
        # Add assistant message to conversation
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        })

        if message.content:
            final_text.append(message.content)

        # Handle tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                try:
                    # Execute tool call
                    result = await self.call_tool(tool_name, tool_args)
                    tool_results.append({"call": tool_name, "result": result})
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content)
                    })
                except Exception as e:
                    error_msg = f"Error calling tool {tool_name}: {e}"
                    final_text.append(f"[{error_msg}]")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })

            # Get next response from OpenAI with tool results
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1000,
                messages=messages,
            )

            final_text.append(response.choices[0].message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries, 'servers' to list servers, or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'servers':
                    self.list_servers()
                    continue
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script> [<path_to_server_script2> ...]")
        print("       python client.py server1.py server2.js server3.py")
        sys.exit(1)
        
    client = MCPClient()
    try:
        # Connect to all provided servers
        server_paths = sys.argv[1:]
        print(server_paths)
        await client.connect_to_multiple_servers(server_paths)
        
        if not client.servers:
            print("Failed to connect to any servers. Exiting.")
            sys.exit(1)
            
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())