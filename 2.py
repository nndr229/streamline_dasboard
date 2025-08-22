import os
import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Define tools for the agent

# Tool 1: Parse payslip PDF to extract net salary


def parse_payslip(file_path: str) -> float:
    """Extract net salary from uploaded payslip PDF using LLM."""
    logger.info(f"Parsing payslip from file: {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"No such file: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        # Log first 200 chars
        logger.info(f"Extracted text from PDF: {text[:200]}...")

        # Enhanced prompt to handle various formats
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template="""
            From the following payslip text, extract the net salary (take-home pay) in INR.
            Look for terms like 'BASIC', 'HRA', 'OTHER ALLOWANCE', or similar.
            Return only the numeric value (e.g., 80000). If no salary is found, return 0.
            \n\n{text}
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(text=text)
        logger.info(f"LLM response for salary: {response}")
        try:
            net_salary = float(response.strip().replace(
                ",", "").replace("INR", ""))
            if net_salary <= 0:
                logger.warning("Extracted net salary is 0 or negative")
            logger.info(f"Extracted net salary: {net_salary}")
            return net_salary
        except ValueError as e:
            logger.error(
                f"Failed to parse net salary from response: {response}, error: {e}")
            return 0.0
    except Exception as e:
        logger.error(f"Error parsing payslip: {e}")
        return 0.0


parse_tool = Tool(
    name="parse_payslip",
    func=parse_payslip,
    description="Parses a payslip PDF file path (string) to extract net monthly salary in INR. Returns a float."
)

# Tool 2: Calculate budget allocations based on salary, loan, and bills


def calculate_budget(inputs: str) -> dict:
    """Calculate budget allocations using 50/30/20 rule, adjusted for debts."""
    try:
        # Handle both string and dict inputs
        if isinstance(inputs, str):
            try:
                inputs_dict = json.loads(inputs)
            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON decode error for inputs: {inputs}, error: {e}")
                return {"error": f"Invalid input format: {str(e)}"}
        else:
            inputs_dict = inputs
        logger.info(f"Parsed inputs: {inputs_dict}")

        net_salary = float(inputs_dict.get("net_salary", 0))
        loan = float(inputs_dict.get("loan", 0))
        bills = float(inputs_dict.get("bills", 0))
        logger.info(
            f"Calculating budget: net_salary={net_salary}, loan={loan}, bills={bills}")

        disposable = net_salary - loan - bills
        if disposable <= 0:
            logger.warning("Disposable income is zero or negative")
            return {"error": "Disposable income is zero or negative. Adjust salary or debts."}

        # Breakdown needs into sub-categories (custom for India)
        needs_breakdown = {
            "Housing/Rent": disposable * 0.5 * 0.35,  # 35% of needs
            "Food/Groceries": disposable * 0.5 * 0.20,
            "Transport": disposable * 0.5 * 0.15,
            "Utilities": disposable * 0.5 * 0.10,
            "Health/Education": disposable * 0.5 * 0.10,
            "Other Needs": disposable * 0.5 * 0.10
        }

        allocations = {
            "Needs": sum(needs_breakdown.values()),
            "Wants": disposable * 0.3,
            "Savings": disposable * 0.2,
            "Needs Breakdown": needs_breakdown
        }
        allocations["Daily"] = {
            k: v / 30 for k, v in allocations.items() if k not in ["Needs Breakdown"]}
        allocations["Weekly"] = {
            k: v / 4 for k, v in allocations.items() if k not in ["Needs Breakdown"]}
        logger.info(f"Budget calculated: {allocations}")
        return allocations
    except Exception as e:
        logger.error(f"Error calculating budget: {e}")
        return {"error": str(e)}


budget_tool = Tool(
    name="calculate_budget",
    func=calculate_budget,
    description="Calculates budget allocations from a JSON string or dict with net_salary (float), loan (float), bills (float)."
)

# Tool 3: Generate visualizations (pie and bar charts)


def generate_graphs(allocations: dict) -> str:
    """Generate pie chart for monthly allocations and bar charts for slabs."""
    try:
        if "error" in allocations:
            logger.error(
                f"Cannot generate graphs due to error in allocations: {allocations['error']}")
            return f"Error: {allocations['error']}"
        logger.info("Generating budget graphs")

        # Pie chart for monthly allocations
        fig1, ax1 = plt.subplots()
        ax1.pie([allocations["Needs"], allocations["Wants"], allocations["Savings"]],
                labels=["Needs", "Wants", "Savings"], autopct="%1.1f%%")
        ax1.set_title("Monthly Budget Allocation")
        st.pyplot(fig1)

        # Detailed needs pie
        fig2, ax2 = plt.subplots()
        needs = allocations["Needs Breakdown"]
        ax2.pie(needs.values(), labels=needs.keys(), autopct="%1.1f%%")
        ax2.set_title("Needs Breakdown")
        st.pyplot(fig2)

        # Bar chart for daily/weekly/monthly
        periods = ["Daily", "Weekly", "Monthly"]
        df_periods = pd.DataFrame({
            "Needs": [allocations["Daily"]["Needs"], allocations["Weekly"]["Needs"], allocations["Needs"]],
            "Wants": [allocations["Daily"]["Wants"], allocations["Weekly"]["Wants"], allocations["Wants"]],
            "Savings": [allocations["Daily"]["Savings"], allocations["Weekly"]["Savings"], allocations["Savings"]]
        }, index=periods)
        fig3, ax3 = plt.subplots()
        df_periods.plot(kind="bar", ax=ax3)
        ax3.set_title("Expenditure and Savings Across Periods")
        ax3.set_ylabel("Amount (INR)")
        st.pyplot(fig3)

        # Slab visualizations (80k to 5L)
        slabs = [80000, 150000, 250000, 350000, 500000]
        slab_allocs = []
        for s in slabs:
            alloc = calculate_budget({"net_salary": s, "loan": 0, "bills": 0})
            if "error" not in alloc:
                slab_allocs.append({
                    "Salary": s,
                    "Needs": alloc["Needs"],
                    "Wants": alloc["Wants"],
                    "Savings": alloc["Savings"]
                })
        if slab_allocs:
            df_slabs = pd.DataFrame(slab_allocs).set_index("Salary")
            fig4, ax4 = plt.subplots()
            df_slabs.plot(kind="bar", stacked=True, ax=ax4)
            ax4.set_title("Budget Slabs from 80k to 5L")
            ax4.set_ylabel("Amount (INR)")
            st.pyplot(fig4)

        return "Graphs generated successfully."
    except Exception as e:
        logger.error(f"Error generating graphs: {e}")
        return f"Error generating graphs: {str(e)}"


graph_tool = Tool(
    name="generate_graphs",
    func=generate_graphs,
    description="Generates budget graphs based on allocations dictionary."
)

# Initialize Langchain Agent
tools = [parse_tool, budget_tool, graph_tool]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit Dashboard App
st.title("Budget Planner Dashboard")

# Upload payslip
uploaded_file = st.file_uploader("Upload your payslip (PDF)", type="pdf")

# Inputs for loan and credit card bills
loan_amount = st.number_input(
    "Enter monthly loan amount (INR)", min_value=0.0, value=0.0)
credit_bills = st.number_input(
    "Enter monthly credit card bills (INR)", min_value=0.0, value=0.0)

# Store results for display
if "budget_results" not in st.session_state:
    st.session_state.budget_results = None

if st.button("Generate Budget Plan and Dashboard"):
    if uploaded_file:
        temp_file_path = "temp_payslip.pdf"
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Run agent with explicit steps
            prompt = f"""
            Step 1: Use parse_payslip tool with file path '{temp_file_path}' to extract Total Eernings.
            Step 2: Use calculate_budget tool with inputs {{ "net_salary": <result from parse_payslip>, "loan": {loan_amount}, "bills": {credit_bills} }}.
            Step 3: Use generate_graphs tool with the budget allocations from calculate_budget.
            """
            logger.info(f"Running agent with prompt: {prompt}")
            response = agent.run(prompt)
            st.session_state.budget_results = response
            st.write("Agent Response:", response)

            # Display extracted salary and budget details
            net_salary = parse_payslip(temp_file_path)
            if net_salary > 0:
                st.write(f"Extracted Net Salary: {net_salary:,.2f} INR")
                budget = calculate_budget(json.dumps(
                    {"net_salary": net_salary, "loan": loan_amount, "bills": credit_bills}))
                if "error" not in budget:
                    st.write("### Budget Allocations")
                    st.write(f"**Monthly Needs**: {budget['Needs']:,.2f} INR")
                    st.write(f"**Monthly Wants**: {budget['Wants']:,.2f} INR")
                    st.write(
                        f"**Monthly Savings**: {budget['Savings']:,.2f} INR")
                    st.write("**Needs Breakdown**:")
                    for k, v in budget["Needs Breakdown"].items():
                        st.write(f"- {k}: {v:,.2f} INR")
                    st.write(
                        f"**Daily Needs**: {budget['Daily']['Needs']:,.2f} INR")
                    st.write(
                        f"**Weekly Needs**: {budget['Weekly']['Needs']:,.2f} INR")
                    st.write(
                        f"**Daily Wants**: {budget['Daily']['Wants']:,.2f} INR")
                    st.write(
                        f"**Weekly Wants**: {budget['Weekly']['Wants']:,.2f} INR")
                    st.write(
                        f"**Daily Savings**: {budget['Daily']['Savings']:,.2f} INR")
                    st.write(
                        f"**Weekly Savings**: {budget['Weekly']['Savings']:,.2f} INR")
            else:
                st.error("Failed to extract a valid net salary from the payslip.")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            st.error(f"Error: {str(e)}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
    else:
        st.error("Please upload a payslip PDF.")

# Debug section
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    if os.path.exists("temp_payslip.pdf"):
        st.write("Temporary payslip file exists.")
    else:
        st.write("No temporary payslip file found.")
    st.write("Available tools:", [tool.name for tool in tools])
    if st.session_state.budget_results:
        st.write("Last Agent Response:", st.session_state.budget_results)
