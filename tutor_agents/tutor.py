"""
Multi-Agent Tutoring Bot with ReAct Architecture and Gemini Function Calling
A sophisticated AI tutoring system where agents reason and act in loops until task completion
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_VERIFICATION = "needs_verification"

@dataclass
class ReActStep:
    """Single step in ReAct reasoning"""
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: str

@dataclass
class Task:
    """Task definition for agents"""
    id: str
    query: str
    assigned_agent: str
    status: TaskStatus
    steps: List[ReActStep]
    final_result: Optional[str] = None
    verification_attempts: int = 0
    max_verification_attempts: int = 3

@dataclass
class Message:
    """Data class for conversation messages"""
    role: str
    content: str
    timestamp: str
    agent: str = "system"
    task_id: Optional[str] = None
    react_steps: List[ReActStep] = None

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Message]] = {}
        self.tasks: Dict[str, Task] = {}
        self.max_history = 15
        self.logger = logging.getLogger(__name__ + '.ConversationManager')
    
    def add_message(self, session_id: str, role: str, content: str, agent: str = "system", 
                   task_id: str = None, react_steps: List[ReActStep] = None):
        """Add a message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            agent=agent,
            task_id=task_id,
            react_steps=react_steps or []
        )
        
        self.conversations[session_id].append(message)
        
        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history*2:]
    
    def create_task(self, query: str, assigned_agent: str) -> str:
        """Create a new task"""
        task_id = f"task_{datetime.now().timestamp()}"
        print(task_id)
        task = Task(
            id=task_id,
            query=query,
            assigned_agent=assigned_agent,
            status=TaskStatus.PENDING,
            steps=[]
        )
        self.tasks[task_id] = task
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus, result: str = None):
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if result:
                self.tasks[task_id].final_result = result

class Tool:
    """Base class for agent tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(__name__ + f'.Tool.{name}')
    
    def get_function_declaration(self) -> Dict[str, Any]:
        """Get the function declaration for Gemini function calling"""
        raise NotImplementedError
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        raise NotImplementedError

class Calculator(Tool):
    """Calculator tool with operator-based approach"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations using operator and operands"
        )
    
    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": "calculator",
            "description": "Performs mathematical calculations. Use for arithmetic operations, evaluating expressions, or any numerical computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator": {
                        "type": "string",
                        "description": "Mathematical operator (+, -, *, /, **, sqrt, sin, cos, tan, log, exp, etc.)",
                        "enum": ["+", "-", "*", "/", "**", "sqrt", "sin", "cos", "tan", "log", "exp", "abs", "round"]
                    },
                    "operand1": {
                        "type": "number",
                        "description": "First operand for the operation"
                    },
                    "operand2": {
                        "type": "number",
                        "description": "Second operand for the operation (not needed for unary operations like sqrt, sin, etc.)"
                    },
                    
                },
                "required": ["operator", "operand1", "operand2"]
            }
        }
    
    def execute(self, operator: str, operand1: float = None, operand2: float = None, expression: str = None) -> Dict[str, Any]:
        """Execute calculation"""
        self.logger.info(f"Executing calculation: {operator} with operands {operand1}, {operand2}")
        
        try:
            import math
            
            if expression:
                # Handle complex expressions
                cleaned = re.sub(r'[^0-9+\-*/().\s^sqrt|sin|cos|tan|log|exp|pi|e]', '', expression)
                cleaned = cleaned.replace('^', '**')
                safe_dict = {
                    "__builtins__": {},
                    "abs": abs, "round": round, "pow": pow,
                    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                    "tan": math.tan, "log": math.log, "exp": math.exp,
                    "pi": math.pi, "e": math.e
                }
                result = eval(cleaned, safe_dict)
                return {
                    "success": True,
                    "result": result,
                    "operation": f"Expression: {expression}",
                    "formatted": f"{expression} = {result}"
                }
            
            # Handle operator-based calculations
            if operator in ["+", "-", "*", "/", "**", "%"]:
                if operand1 is None or operand2 is None:
                    raise ValueError(f"Binary operator {operator} requires two operands")
                
                if operator == "+":
                    result = operand1 + operand2
                elif operator == "-":
                    result = operand1 - operand2
                elif operator == "*":
                    result = operand1 * operand2
                elif operator == "/":
                    if operand2 == 0:
                        raise ValueError("Division by zero")
                    result = operand1 / operand2
                elif operator == "**":
                    result = operand1 ** operand2
                elif operator == "%":
                    result = operand1 % operand2
                    
            elif operator in ["sqrt", "sin", "cos", "tan", "log", "exp", "abs", "round"]:
                if operand1 is None:
                    raise ValueError(f"Unary operator {operator} requires one operand")
                
                if operator == "sqrt":
                    result = math.sqrt(operand1)
                elif operator == "sin":
                    result = math.sin(operand1)
                elif operator == "cos":
                    result = math.cos(operand1)
                elif operator == "tan":
                    result = math.tan(operand1)
                elif operator == "log":
                    result = math.log(operand1)
                elif operator == "exp":
                    result = math.exp(operand1)
                elif operator == "abs":
                    result = abs(operand1)
                elif operator == "round":
                    result = round(operand1, int(operand2) if operand2 else 0)
            else:
                raise ValueError(f"Unknown operator: {operator}")
            
            return {
                "success": True,
                "result": result,
                "operation": f"{operator}({operand1}, {operand2})" if operand2 is not None else f"{operator}({operand1})",
                "formatted": f"{operand1} {operator} {operand2} = {result}" if operator in ["+", "-", "*", "/", "**", "%"] else f"{operator}({operand1}) = {result}"
            }
            
        except Exception as e:
            self.logger.error(f"Calculator error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "operation": f"{operator}({operand1}, {operand2})"
            }

class PhysicsConstants(Tool):
    """Physics constants lookup tool"""
    
    def __init__(self):
        super().__init__(
            name="physics_constants",
            description="Looks up fundamental physics constants"
        )
        
        self.constants = {
            "speed_of_light": {"value": 299792458, "unit": "m/s", "symbol": "c"},
            "planck_constant": {"value": 6.62607015e-34, "unit": "J⋅s", "symbol": "h"},
            "gravitational_constant": {"value": 6.67430e-11, "unit": "m³/kg⋅s²", "symbol": "G"},
            "electron_mass": {"value": 9.1093837015e-31, "unit": "kg", "symbol": "mₑ"},
            "proton_mass": {"value": 1.67262192369e-27, "unit": "kg", "symbol": "mₚ"},
            "avogadro_number": {"value": 6.02214076e23, "unit": "mol⁻¹", "symbol": "Nₐ"},
            "boltzmann_constant": {"value": 1.380649e-23, "unit": "J/K", "symbol": "k"},
            "elementary_charge": {"value": 1.602176634e-19, "unit": "C", "symbol": "e"}
        }
    
    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": "physics_constants",
            "description": "Looks up fundamental physics constants and their values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "constant_name": {
                        "type": "string",
                        "description": "Name of the physics constant to look up",
                        "enum": list(self.constants.keys())
                    }
                },
                "required": ["constant_name"]
            }
        }
    
    def execute(self, constant_name: str) -> Dict[str, Any]:
        """Look up physics constant"""
        self.logger.info(f"Looking up constant: {constant_name}")
        constant_name = constant_name.lower().replace(" ", "_")
        
        if constant_name in self.constants:
            return {
                "success": True,
                "constant": self.constants[constant_name],
                "name": constant_name
            }
        else:
            return {
                "success": False,
                "error": f"Constant '{constant_name}' not found",
                "available": list(self.constants.keys())
            }

class ChemistryData(Tool):
    """Chemistry data lookup tool"""
    
    def __init__(self):
        super().__init__(
            name="chemistry_data",
            description="Provides chemical element information"
        )
        
        self.elements = {
            "hydrogen": {"symbol": "H", "atomic_number": 1, "atomic_mass": 1.008},
            "helium": {"symbol": "He", "atomic_number": 2, "atomic_mass": 4.003},
            "carbon": {"symbol": "C", "atomic_number": 6, "atomic_mass": 12.011},
            "nitrogen": {"symbol": "N", "atomic_number": 7, "atomic_mass": 14.007},
            "oxygen": {"symbol": "O", "atomic_number": 8, "atomic_mass": 15.999},
            "sodium": {"symbol": "Na", "atomic_number": 11, "atomic_mass": 22.990},
            "iron": {"symbol": "Fe", "atomic_number": 26, "atomic_mass": 55.845},
            "gold": {"symbol": "Au", "atomic_number": 79, "atomic_mass": 196.97}
        }
    
    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": "chemistry_data",
            "description": "Looks up chemical element information including atomic number, mass, and symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Name or symbol of the chemical element"
                    }
                },
                "required": ["element"]
            }
        }
    
    def execute(self, element: str) -> Dict[str, Any]:
        """Look up element information"""
        self.logger.info(f"Looking up element: {element}")
        query = element.lower().strip()
        
        for name, data in self.elements.items():
            if query == name or query == data["symbol"].lower():
                return {
                    "success": True,
                    "element": data,
                    "name": name.capitalize()
                }
        
        return {
            "success": False,
            "error": f"Element '{element}' not found",
            "available": list(self.elements.keys())
        }

class BaseAgent:
    """Base agent class with ReAct reasoning capability and proper Gemini function calling"""
    
    def __init__(self, name: str, role: str, model_name: str = "gemini-2.0-flash"):
        self.name = name
        self.role = role
        self.model = model_name
        self.tools: List[Tool] = []
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.logger = logging.getLogger(__name__ + f'.Agent.{name}')
        self.max_react_steps = 10
        self.max_function_calls = 5  # Prevent infinite function calling loops
    
    def add_tool(self, tool: Tool):
        """Add a tool to this agent"""
        self.logger.info(f"Adding tool: {tool.name}")
        self.tools.append(tool)
    
    def get_tools_config(self) -> Optional[types.GenerateContentConfig]:
        """Get tools configuration for function calling"""
        if not self.tools:
            return None
        
        function_declarations = [tool.get_function_declaration() for tool in self.tools]
        tools = types.Tool(function_declarations=function_declarations)
        return types.GenerateContentConfig(tools=[tools])
    
    def llm_call_with_functions(self, contents: List[types.Content]) -> Tuple[str, List[types.Content]]:
        """
        Make an LLM call that properly handles function calling.
        Returns the final text response and the complete conversation history.
        """
        config = self.get_tools_config()
        current_contents = contents.copy()
        function_call_count = 0
        final_result = None
        
        while function_call_count < self.max_function_calls:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    config=config,
                    contents=current_contents
                )
                
                
                # Check if the response contains a function call
                if (response.candidates and 
                    response.candidates[0].content.parts and 
                    hasattr(response.candidates[0].content.parts[0], 'function_call') and
                    response.candidates[0].content.parts[0].function_call):
                    
                    function_call = response.candidates[0].content.parts[0].function_call
                    self.logger.info(f"Function call requested: {function_call.name}")
                    print(function_call)
                    # Add the model's function call to the conversation
                    current_contents.append(
                        types.Content(
                            role="model", 
                            parts=[types.Part(function_call=function_call)]
                        )
                    )
                    
                    # Execute the function
                    result = self.execute_function_call(function_call.name, function_call.args)
                    final_result = result
                    # Create function response part
                    function_response_part = types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result}
                    )
                    
                    # Add the function result to the conversation
                    current_contents.append(
                        types.Content(
                            role="user", 
                            parts=[function_response_part]
                        )
                    )
                    
                    function_call_count += 1
                    continue
                
                # If no function call, return the text response
                if response.text:
                    return f"Result from calculator tool: {final_result.get('formatted')}" if final_result else response.text, current_contents
                else:
                    return "No response generated", current_contents
                    
            except Exception as e:
                self.logger.error(f"LLM call error: {str(e)}")
                return f"Error in LLM call: {str(e)}", current_contents
        
        return "Maximum function calls reached without final response", current_contents
    
    def llm_call(self, prompt: str, use_tools: bool = False) -> str:
        """Simple LLM call - maintains backward compatibility"""
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        
        if use_tools and self.tools:
            response_text, _ = self.llm_call_with_functions(contents)
            return response_text
        else:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents
                )
                return response.text.strip() if response.text else "No response generated"
            except Exception as e:
                self.logger.error(f"LLM call error: {str(e)}")
                return f"Error in LLM call: {str(e)}"
    
    def execute_function_call(self, function_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call"""
        tool = next((t for t in self.tools if t.name == function_name), None)
        if tool:
            try:
                return tool.execute(**function_args)
            except Exception as e:
                self.logger.error(f"Tool execution error: {str(e)}")
                return {"success": False, "error": f"Tool execution failed: {str(e)}"}
        else:
            return {"success": False, "error": f"Tool {function_name} not found"}
    
    def react_step(self, query: str, step_number: int, previous_steps: List[ReActStep]) -> ReActStep:
        """Execute one ReAct step: Think -> Act -> Observe"""
        
        # Create context from previous steps
        context = f"Original Query: {query}\n\n"
        if previous_steps:
            context += "Previous Steps:\n"
            for step in previous_steps:
                context += f"Step {step.step_number}:\n"
                context += f"Thought: {step.thought}\n"
                context += f"Action: {step.action}\n"
                context += f"Observation: {step.observation}\n\n"
        
        # Think: Generate reasoning
        think_prompt = f"""
        {context}
        
        You are {self.role}. 
        
        Based on the query and previous steps, what should you think about next?
        Consider what information you need, what calculations to perform, or what conclusion to draw.
        Respond with your reasoning/thinking for this step. Be specific about what you need to do next.
        """
        
        thought = self.llm_call(think_prompt)
        
        # print(thought)
        # Act: Decide on action and potentially execute tools
        act_prompt = f"""
        CONTEXT: {context}

        Current Thought: {thought}
        
        You are following the ReAct (Reasoning and Acting) pattern. Based on your current thinking, you need to decide on the next action.
        
        You have these options:
        1. Use a tool - If you need to gather information, perform calculations, or execute an action
        2. Conclude - If you have enough information to provide a final answer
        3. Continue reasoning - If you need to think more but don't need tools yet
        
        Available tools: {[f"{tool.name}: {tool.description}" for tool in self.tools]}
        
        IMPORTANT: 
        - If you need to use a tool, simply call the appropriate function call
        - If you're ready to conclude, respond with "CONCLUDE: [your reasoning for concluding]"
        - If you need to continue reasoning, respond with "CONTINUE: [what you need to think about next]"
        
        You have to use the given Tools if there is any arithmetic operation to be performed or chemical element to be found or physics constant to be found.
        What action do you take?
        """
        # print(act_prompt)
        # Use function calling for the action step
        contents = [types.Content(role="user", parts=[types.Part(text=act_prompt)])]
        action_response, conversation_history = self.llm_call_with_functions(contents)

        # print(conversation_history)
        print(action_response)
        # Parse the action from the response and conversation history
        action = "continue"  # Default action
        action_input = {}
        observation = action_response
        
        # Check if any functions were called during this step
        function_calls_made = []
        for content in conversation_history:
            if content.role == "model" and content.parts:
                for part in content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls_made.append({
                            'name': part.function_call.name,
                            'args': part.function_call.args
                        })
        
        if function_calls_made:
            # Function was called - this is the action
            action = function_calls_made[-1]['name']
            action_input = function_calls_made[-1]['args']
            observation = f"Tool '{action}' executed with args {action_input}. Result: {action_response}"
        else:
            # No function called - parse the text response for action intent
            response_lower = action_response.lower()
            if action_response.startswith("CONCLUDE:") or "conclude" in response_lower:
                action = "conclude"
                observation = f"Agent decided to conclude: {action_response}"
            elif action_response.startswith("CONTINUE:") or "continue" in response_lower:
                action = "continue"
                observation = f"Agent decided to continue reasoning: {action_response}"
            else:
                # Try to infer action from keywords
                if any(tool.name.lower() in response_lower for tool in self.tools):
                    action = "tool_mentioned"
                    observation = f"Agent mentioned using tools but didn't call them: {action_response}"
                elif any(keyword in response_lower for keyword in ["done", "finished", "final", "answer", "complete"]):
                    action = "conclude"
                    observation = f"Agent indicated completion: {action_response}"
                else:
                    action = "continue"
                    observation = f"Agent response: {action_response}"
        
        return ReActStep(
            step_number=step_number,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            timestamp=datetime.now().isoformat()
        )
    
    def execute_task_with_react(self, task: Task) -> Tuple[bool, str, List[ReActStep]]:
        """Execute a task using ReAct reasoning loop"""
        self.logger.info(f"Executing task with ReAct: {task.query}")
        
        steps = []
        
        for step_num in range(1, self.max_react_steps + 1):
            self.logger.info(f"Executing ReAct step {step_num}")
            
            step = self.react_step(task.query, step_num, steps)
            steps.append(step)
            
            # Check if we should conclude
            if step.action == "conclude":
                
                # Generate final answer
                final_prompt = f"""
                Original Query: {task.query}
                
                ReAct Steps Taken:
                {chr(10).join([f"Step {s.step_number}: Thought: {s.thought}, Action: {s.action}, Observation: {s.observation}" for s in steps])}
                
                Based on all the reasoning and actions taken, provide a comprehensive final answer to the query.
                Be clear, accurate, and educational in your response.
                """
                
                final_answer = self.llm_call(final_prompt)
                return True, final_answer, steps
        
        # If we reach max steps without concluding
        final_answer = "I was unable to complete the task within the maximum number of reasoning steps."
        return False, final_answer, steps

class MathAgent(BaseAgent):
    """Specialized agent for mathematics"""
    
    def __init__(self):
        super().__init__(
            name="Math Agent",
            role="an expert mathematics tutor specializing in algebra, calculus, geometry, and problem solving"
        )
        self.add_tool(Calculator())

class PhysicsAgent(BaseAgent):
    """Specialized agent for physics"""
    
    def __init__(self):
        super().__init__(
            name="Physics Agent", 
            role="an expert physics tutor specializing in mechanics, thermodynamics, and electromagnetic theory"
        )
        self.add_tool(PhysicsConstants())
        self.add_tool(Calculator())

class ChemistryAgent(BaseAgent):
    """Specialized agent for chemistry"""
    
    def __init__(self):
        super().__init__(
            name="Chemistry Agent",
            role="an expert chemistry tutor specializing in chemical reactions, periodic table, and molecular structure"
        )
        self.add_tool(ChemistryData())
        self.add_tool(Calculator())

class HistoryAgent(BaseAgent):
    """Specialized agent for history"""
    
    def __init__(self):
        super().__init__(
            name="History Agent",
            role="an expert history tutor specializing in world history, historical analysis, and chronological understanding"
        )

class MainTutorAgent(BaseAgent):
    """Main orchestrating tutor agent"""
    
    def __init__(self):
        super().__init__(
            name="Main Tutor Agent",
            role="an AI tutoring coordinator that analyzes queries and delegates to appropriate specialist agents"
        )
        
        # Initialize specialist agents
        self.specialist_agents = {
            "math": MathAgent(),
            "physics": PhysicsAgent(), 
            "chemistry": ChemistryAgent(),
            "history": HistoryAgent()
        }
    
    def classify_query(self, query: str) -> str:
        """Use LLM to classify the query subject"""
        self.logger.info(f"Classifying query: {query[:100]}...")
        
        classification_prompt = f"""
        Analyze this student question and classify it into one of these subjects:
        - math: Mathematics, algebra, calculus, geometry, arithmetic, statistics
        - physics: Physics concepts, mechanics, thermodynamics, electromagnetic theory
        - chemistry: Chemical reactions, elements, compounds, periodic table
        - history: Historical events, dates, civilizations, wars, historical figures
        - none: General questions not specific to any subject
        
        Student Question: "{query}"
        
        Consider the main focus and content of the question. Respond with just the subject name (math/physics/chemistry/history/none).
        if the question spans multiple subjects, choose the primary subject.
        If the question is general and not specific to any subject, respond with 'none'.
        """
        
        subject = self.llm_call(classification_prompt).lower().strip()
        
        if subject not in self.specialist_agents and subject != "none":
            subject = "none"  # Default to none for general queries
        
        self.logger.info(f"Classified as: {subject}")
        return subject
    
    def handle_general_query(self, query: str) -> str:
        """Handle general queries directly with Gemini"""
        self.logger.info("Handling general query with Gemini")
        
        prompt = f"""
        You are a helpful AI tutor. Please provide a clear and informative response to this general question:
        
        {query}
        """
        
        response = self.llm_call(prompt)
        return response
    
    def verify_result(self, original_query: str, agent_result: str, subject: str) -> Tuple[bool, str]:
        """Use LLM to verify if the agent's result is satisfactory"""
        self.logger.info("Verifying agent result with LLM")
        
        verification_prompt = f"""
        You are verifying the quality and correctness of a tutoring response.
        
        Original Student Question: "{original_query}"
        Subject Area: {subject}
        Agent's Response: "{agent_result}"
        
        Evaluate this response based on:
        1. Correctness - Is the information accurate?
        2. Completeness - Does it fully address the question?
        3. Clarity - Is it well explained for a student?
        4. Educational Value - Does it help the student learn?
        
        Respond in this format:
        VERDICT: [APPROVED/REJECTED]
        REASON: [Brief explanation of your decision]
        FEEDBACK: [If rejected, what needs improvement]
        """
        
        verification_response = self.llm_call(verification_prompt)
        
        # Parse verification response
        lines = verification_response.split('\n')
        verdict = "REJECTED"  # Default to rejected for safety
        reason = "Verification failed"
        
        for line in lines:
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        
        approved = verdict.upper() == "APPROVED"
        self.logger.info(f"Verification result: {verdict} - {reason}")
        
        return approved, reason
    
    def process_query(self, query: str, session_id: str, conversation_manager: ConversationManager) -> Dict[str, Any]:
        """Main query processing with ReAct loop and verification"""
        self.logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Step 1: Classify the query using LLM
            subject = self.classify_query(query)
            
            # Handle general queries directly
            if subject == "none":
                response = self.handle_general_query(query)
                return {
                    "success": True,
                    "response": response,
                    "subject": "general",
                    "agent": self.name,
                    "verification_passed": True
                }
            
            specialist_agent = self.specialist_agents[subject]
            
            # Step 2: Create task for specialist agent
            task_id = conversation_manager.create_task(query, specialist_agent.name)
            task = conversation_manager.get_task(task_id)
            
            max_attempts = 3
            attempt = 1
            
            while attempt <= max_attempts:
                self.logger.info(f"Attempt {attempt} - Delegating to {specialist_agent.name}")
                
                # Step 3: Specialist agent executes task using ReAct
                task.status = TaskStatus.IN_PROGRESS
                success, result, react_steps = specialist_agent.execute_task_with_react(task)
                
                if not success:
                    return {
                        "success": False,
                        "error": "Specialist agent failed to complete task",
                        "subject": subject,
                        "agent": specialist_agent.name,
                        "attempt": attempt,
                        "react_steps": [asdict(step) for step in react_steps]
                    }
                
                # Step 4: Main tutor verifies the result using LLM
                self.logger.info("Verifying result with main tutor")
                approved, verification_reason = self.verify_result(query, result, subject)
                
                if approved:
                    # Result approved - complete the task
                    task.status = TaskStatus.COMPLETED
                    task.final_result = result
                    task.steps = react_steps
                    
                    return {
                        "success": True,
                        "response": result,
                        "subject": subject,
                        "agent": specialist_agent.name,
                        "task_id": task_id,
                        "verification_passed": True,
                        "verification_reason": verification_reason,
                        "attempts": attempt,
                        "react_steps": [asdict(step) for step in react_steps],
                        "react_summary": f"Completed in {len(react_steps)} reasoning steps"
                    }
                else:
                    # Result rejected - try again if attempts remain
                    task.verification_attempts += 1
                    self.logger.warning(f"Result rejected: {verification_reason}")
                    
                    if attempt < max_attempts:
                        # Provide feedback to specialist agent for next attempt
                        feedback_query = f"""
                        Original Query: {query}
                        Previous Attempt Result: {result}
                        Verification Feedback: {verification_reason}
                        
                        Please improve your response based on the feedback and try again.
                        """
                        task.query = feedback_query
                        attempt += 1
                    else:
                        # Max attempts reached
                        task.status = TaskStatus.FAILED
                        return {
                            "success": False,
                            "error": "Maximum verification attempts reached",
                            "subject": subject,
                            "agent": specialist_agent.name,
                            "task_id": task_id,
                            "last_result": result,
                            "verification_reason": verification_reason,
                            "attempts": attempt,
                            "react_steps": [asdict(step) for step in react_steps]
                        }
            
        except Exception as e:
            self.logger.error(f"Error in main tutor processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }

# Initialize global instances
conversation_manager = ConversationManager()
main_tutor = MainTutorAgent()

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with ReAct architecture"""
    logger.info("Received chat request")
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Empty message"
            }), 400
        
        # Process query with main tutor using ReAct architecture
        result = main_tutor.process_query(query, session_id, conversation_manager)
        
        if result.get("success"):
            # Add user message
            conversation_manager.add_message(session_id, "user", query)
            
            # Add assistant response with ReAct steps
            conversation_manager.add_message(
                session_id, 
                "assistant", 
                result["response"], 
                result.get("agent", "Main Tutor"),
                result.get("task_id"),
                [ReActStep(**step) for step in result.get("react_steps", [])]
            )
            
            logger.info(f"Successfully processed query with {result.get('agent')} in {result.get('attempts', 1)} attempt(s)")
            
            # ENHANCED: Always include task_id in response, even for general queries
            if not result.get("task_id") and result.get("subject") == "general":
                # Create a dummy task_id for general queries for consistency
                result["task_id"] = f"general_{datetime.now().timestamp()}"
            
            # Add ReAct summary to response for frontend
            if result.get("react_steps"):
                result["react_summary"] = f"Completed reasoning in {len(result['react_steps'])} steps"
                result["reasoning_trace"] = [
                    {
                        "step": step["step_number"],
                        "thought": step["thought"][:100] + "..." if len(step["thought"]) > 100 else step["thought"],
                        "action": step["action"],
                        "success": "error" not in step.get("observation", "").lower()
                    }
                    for step in result["react_steps"]
                ]
        else:
            logger.error(f"Failed to process query: {result.get('error')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/api/history/<session_id>')
def get_history(session_id):
    """Get conversation history for a session"""
    logger.info(f"Getting history for session {session_id}")
    try:
        history = conversation_manager.conversations.get(session_id, [])
        logger.debug(f"Found {len(history)} messages")
        
        # Convert ReActStep objects to dictionaries for JSON serialization
        serialized_history = []
        for msg in history:
            msg_dict = asdict(msg)
            if msg_dict.get("react_steps"):
                msg_dict["react_steps"] = [asdict(step) for step in msg.react_steps]
            serialized_history.append(msg_dict)
        
        return jsonify({
            "success": True,
            "history": serialized_history
        })
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve history"
        }), 500
    

class TaskJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

def serialize_task(task):
    """Convert task object to JSON-serializable dict"""
    task_dict = asdict(task)
    
    # Convert steps to dicts
    task_dict["steps"] = [asdict(step) for step in task.steps]
    
    # Convert to JSON string and back to handle enums
    json_str = json.dumps(task_dict, cls=TaskJSONEncoder)
    return json.loads(json_str)

@app.route('/api/agents')
def get_agents():
    """Get information about available agents and their tools"""
    logger.info("Getting agent information")
    try:
        agents_info = {}
        for name, agent in main_tutor.specialist_agents.items():
            agents_info[name] = {
                "name": agent.name,
                "role": agent.role,
                "tools": [{"name": tool.name, "description": tool.description} for tool in agent.tools],
                "function_declarations": [tool.get_function_declaration() for tool in agent.tools]
            }
        
        return jsonify({
            "success": True,
            "agents": agents_info,
            "main_agent": {
                "name": main_tutor.name,
                "role": main_tutor.role
            },
            "architecture": "ReAct (Reasoning and Acting)"
        })
    except Exception as e:
        logger.error(f"Agents info error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get agents info"
        }), 500

@app.route('/api/tasks/<task_id>')
def get_task(task_id):
    """Get detailed information about a specific task"""
    logger.info(f"Getting task details for {task_id}")
    try:
        task = conversation_manager.get_task(task_id)
        if not task:
            return jsonify({
                "success": False,
                "error": "Task not found"
            }), 404
        
        task_dict = serialize_task(task)
        
        return jsonify({
            "success": True,
            "task": task_dict
        })
    except Exception as e:
        logger.error(f"Task details error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get task details"
        }), 500


@app.route('/api/react-trace/<task_id>')
def get_react_trace(task_id):
    """Get the ReAct reasoning trace for a specific task"""
    logger.info(f"Getting ReAct trace for task {task_id}")
    try:
        task = conversation_manager.get_task(task_id)
        if not task:
            return jsonify({
                "success": False,
                "error": "Task not found"
            }), 404
        
        trace = {
            "task_id": task_id,
            "query": task.query,
            "agent": task.assigned_agent,
            "status": task.status.value,
            "total_steps": len(task.steps),
            "steps": [
                {
                    "step_number": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": step.observation,
                    "timestamp": step.timestamp
                }
                for step in task.steps
            ],
            "final_result": task.final_result,
            "verification_attempts": task.verification_attempts
        }
        
        return jsonify({
            "success": True,
            "trace": trace
        })
    except Exception as e:
        logger.error(f"ReAct trace error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get ReAct trace"
        }), 500

@app.route('/api/clear-session/<session_id>', methods=['POST'])
def clear_session(session_id):
    """Clear conversation history for a session"""
    logger.info(f"Clearing session {session_id}")
    try:
        if session_id in conversation_manager.conversations:
            del conversation_manager.conversations[session_id]
        
        # Also clear related tasks
        tasks_to_remove = []
        for task_id, task in conversation_manager.tasks.items():
            if task_id.startswith(f"task_{session_id}"):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del conversation_manager.tasks[task_id]
        
        return jsonify({
            "success": True,
            "message": f"Session {session_id} cleared"
        })
    except Exception as e:
        logger.error(f"Clear session error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to clear session"
        }), 500

@app.route('/api/statistics')
def get_statistics():
    """Get system statistics"""
    logger.info("Getting system statistics")
    try:
        total_sessions = len(conversation_manager.conversations)
        total_messages = sum(len(msgs) for msgs in conversation_manager.conversations.values())
        total_tasks = len(conversation_manager.tasks)
        
        task_status_counts = {}
        for task in conversation_manager.tasks.values():
            status = task.status.value
            task_status_counts[status] = task_status_counts.get(status, 0) + 1
        
        agent_usage = {}
        for task in conversation_manager.tasks.values():
            agent = task.assigned_agent
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return jsonify({
            "success": True,
            "statistics": {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_tasks": total_tasks,
                "task_status_counts": task_status_counts,
                "agent_usage": agent_usage,
                "architecture": "Multi-Agent ReAct",
                "max_react_steps": main_tutor.specialist_agents["math"].max_react_steps
            }
        })
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to get statistics"
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    try:
        # Test LLM connectivity
        test_response = main_tutor.llm_call("Hello", use_tools=False)
        llm_status = "healthy" if test_response and "error" not in test_response.lower() else "unhealthy"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents_loaded": len(main_tutor.specialist_agents),
            "react_enabled": True,
            "function_calling_enabled": True,
            "llm_status": llm_status,
            "total_tools": sum(len(agent.tools) for agent in main_tutor.specialist_agents.values())
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Multi-Agent Tutoring Bot with ReAct architecture on port {port}")
    logger.info(f"Loaded {len(main_tutor.specialist_agents)} specialist agents")
    logger.info("Architecture: ReAct (Reasoning and Acting in Language Models)")
    app.run(host='0.0.0.0', port=port, debug=False)
