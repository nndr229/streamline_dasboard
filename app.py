import os
import streamlit as st
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

# Tool: Calculate budget allocations


def calculate_budget(inputs: str) -> dict:
    """Calculate budget allocations using 50/30/20 rule, adjusted for family size and debts."""
    try:
        inputs_dict = json.loads(inputs) if isinstance(inputs, str) else inputs
        net_salary = float(inputs_dict.get("net_salary", 0))
        loan = float(inputs_dict.get("loan", 0))
        bills = float(inputs_dict.get("bills", 0))
        family_size = int(inputs_dict.get("family_size", 1))
        logger.info(
            f"Calculating budget: net_salary={net_salary}, loan={loan}, bills={bills}, family_size={family_size}")

        # Adjust needs based on family size (10% increase per additional member)
        needs_factor = 0.5 * (1 + 0.1 * (family_size - 1))
        disposable = net_salary - loan - bills
        if disposable <= 0:
            logger.warning("Disposable income is zero or negative")
            return {"error": "Disposable income is zero or negative. Adjust inputs."}

        # Breakdown needs into sub-categories (custom for India)
        needs_breakdown = {
            "Housing/Rent": disposable * needs_factor * 0.35,
            "Food/Groceries": disposable * needs_factor * 0.25,
            "Transport": disposable * needs_factor * 0.15,
            "Utilities": disposable * needs_factor * 0.10,
            "Health/Education": disposable * needs_factor * 0.10,
            "Other Needs": disposable * needs_factor * 0.05
        }

        wants_factor = 0.3 / (1 + 0.05 * (family_size - 1))
        savings_factor = 0.2 / (1 + 0.05 * (family_size - 1))

        allocations = {
            "Needs": sum(needs_breakdown.values()),
            "Wants": disposable * wants_factor,
            "Savings": disposable * savings_factor,
            "Needs Breakdown": needs_breakdown
        }
        # Calculate daily and weekly allocations only for numeric fields
        numeric_keys = ["Needs", "Wants", "Savings"]
        allocations["Daily"] = {k: allocations[k] / 30 for k in numeric_keys}
        allocations["Weekly"] = {k: allocations[k] / 4 for k in numeric_keys}
        logger.info(f"Budget calculated: {allocations}")
        return allocations
    except Exception as e:
        logger.error(f"Error calculating budget: {e}")
        return {"error": str(e)}


budget_tool = Tool(
    name="calculate_budget",
    func=calculate_budget,
    description="Calculates budget allocations from a JSON string or dict with net_salary (float), loan (float), bills (float), family_size (int)."
)

# Tool: Generate visualizations


def generate_graphs(allocations_input: any) -> str:
    """Generate a 2x2 grid of vivid graphs for budget allocations."""
    try:
        # Handle if allocations is a string (e.g., JSON string from agent)
        if isinstance(allocations_input, str):
            try:
                allocations = json.loads(allocations_input)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse allocations string: {allocations_input}, error: {e}")
                return f"Error parsing allocations: {str(e)}"
        else:
            allocations = allocations_input

        if "error" in allocations:
            logger.error(f"Cannot generate graphs: {allocations['error']}")
            return f"Error: {allocations['error']}"
        logger.info("Generating budget graphs")

        # Ensure all values are floats (in case of serialization issues)
        allocations["Needs"] = float(allocations["Needs"])
        allocations["Wants"] = float(allocations["Wants"])
        allocations["Savings"] = float(allocations["Savings"])
        allocations["Daily"]["Needs"] = float(allocations["Daily"]["Needs"])
        allocations["Daily"]["Wants"] = float(allocations["Daily"]["Wants"])
        allocations["Daily"]["Savings"] = float(
            allocations["Daily"]["Savings"])
        allocations["Weekly"]["Needs"] = float(allocations["Weekly"]["Needs"])
        allocations["Weekly"]["Wants"] = float(allocations["Weekly"]["Wants"])
        allocations["Weekly"]["Savings"] = float(
            allocations["Weekly"]["Savings"])
        for k in allocations["Needs Breakdown"]:
            allocations["Needs Breakdown"][k] = float(
                allocations["Needs Breakdown"][k])

        # Create a 2x2 grid for graphs
        col1, col2 = st.columns(2)

        with col1:
            # Graph 1: Monthly Budget Allocation Pie
            fig1, ax1 = plt.subplots()
            ax1.pie([allocations["Needs"], allocations["Wants"], allocations["Savings"]],
                    labels=["Needs", "Wants", "Savings"],
                    autopct="%1.1f%%",
                    colors=['#ff9999', '#66b3ff', '#99ff99'])
            ax1.set_title("Monthly Budget Allocation")
            st.pyplot(fig1)

            # Graph 3: Expenditure Across Periods Bar
            periods = ["Daily", "Weekly", "Monthly"]
            df_periods = pd.DataFrame({
                "Needs": [allocations["Daily"]["Needs"], allocations["Weekly"]["Needs"], allocations["Needs"]],
                "Wants": [allocations["Daily"]["Wants"], allocations["Weekly"]["Wants"], allocations["Wants"]],
                "Savings": [allocations["Daily"]["Savings"], allocations["Weekly"]["Savings"], allocations["Savings"]]
            }, index=periods)
            fig3, ax3 = plt.subplots()
            df_periods.plot(kind="bar", ax=ax3, color=[
                            '#ff9999', '#66b3ff', '#99ff99'])
            ax3.set_title("Expenditure Across Periods")
            ax3.set_ylabel("Amount (INR)")
            st.pyplot(fig3)

        with col2:
            # Graph 2: Needs Breakdown Pie
            fig2, ax2 = plt.subplots()
            needs = allocations["Needs Breakdown"]
            ax2.pie(needs.values(), labels=needs.keys(), autopct="%1.1f%%",
                    colors=['#ffcc99', '#99ffcc', '#cc99ff', '#ff99cc', '#99ccff', '#ccff99'])
            ax2.set_title("Needs Breakdown")
            st.pyplot(fig2)

            # Graph 4: Budget Slabs Stacked Bar
            slabs = [80000, 150000, 250000, 350000, 500000]
            slab_allocs = []
            for s in slabs:
                alloc = calculate_budget(
                    {"net_salary": s, "loan": 0, "bills": 0, "family_size": 1})
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
                df_slabs.plot(kind="bar", stacked=True, ax=ax4,
                              color=['#ff9999', '#66b3ff', '#99ff99'])
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
    description="Generates a 2x2 grid of budget graphs based on allocations dictionary or JSON string."
)

# Initialize Langchain Agent
tools = [budget_tool, graph_tool]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit Dashboard App
st.title("Payslip Budget Planner Dashboard")

# Form for input fields
with st.form(key="payslip_form"):
    st.header("Payslip and Budget Inputs")
    in_hand_pay = st.number_input(
        "In-Hand Pay (Net Salary, INR)", min_value=0.0, value=0.0)
    pf_deduction = st.number_input(
        "Provident Fund (PF) Deduction (INR)", min_value=0.0, value=0.0)
    tax_percentage = st.number_input(
        "Tax Percentage (%)", min_value=0.0, max_value=100.0, value=0.0)
    loan_amount = st.number_input(
        "Monthly Loan Amount (INR)", min_value=0.0, value=0.0)
    credit_bills = st.number_input(
        "Monthly Credit Card Bills (INR)", min_value=0.0, value=0.0)
    family_size = st.number_input("Family Size", min_value=1, value=1, step=1)
    submit_button = st.form_submit_button(label="Generate Dashboard")

# Store results in session state
if "budget_results" not in st.session_state:
    st.session_state.budget_results = None

if submit_button:
    # Calculate net salary after tax
    tax_amount = in_hand_pay * (tax_percentage / 100)
    net_salary = in_hand_pay - pf_deduction - tax_amount
    if net_salary <= 0:
        st.error("Net salary is zero or negative. Adjust inputs.")
    else:
        st.write(f"### Calculated Net Salary: {net_salary:,.2f} INR")

        # Run agent to calculate budget and generate graphs
        try:
            inputs = json.dumps({
                "net_salary": net_salary,
                "loan": loan_amount,
                "bills": credit_bills,
                "family_size": family_size
            })
            prompt = f"""
            Step 1: Use calculate_budget tool with inputs {inputs}.
            Step 2: Use generate_graphs tool with the budget allocations from calculate_budget.
            """
            logger.info(f"Running agent with prompt: {prompt}")
            response = agent.run(prompt)
            st.session_state.budget_results = response
            st.write("Agent Response:", response)

            # Display budget details
            budget = calculate_budget(inputs)
            if "error" not in budget:
                st.write("### Budget Allocations")
                st.write(f"**Monthly Needs**: {budget['Needs']:,.2f} INR")
                st.write(f"**Monthly Wants**: {budget['Wants']:,.2f} INR")
                st.write(f"**Monthly Savings**: {budget['Savings']:,.2f} INR")
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
                st.error(f"Budget calculation failed: {budget['error']}")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            st.error(f"Error: {str(e)}")

# Debug section
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    st.write("Available tools:", [tool.name for tool in tools])
    if st.session_state.budget_results:
        st.write("Last Agent Response:", st.session_state.budget_results)
