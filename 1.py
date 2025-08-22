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

load_dotenv()

# Set up Gemini LLM (requires GOOGLE_API_KEY in .env)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Define tools for the agent

# Tool 1: Parse payslip PDF to extract net salary


def parse_payslip(file_path):
    """Extract net salary from uploaded payslip PDF using LLM."""
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="From the following payslip text, extract the net salary (take-home pay) in INR. Respond with just the number:\n\n{text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(text=text)
    try:
        net_salary = float(response.strip().replace(",", ""))
    except ValueError:
        net_salary = 0.0
    return net_salary


parse_tool = Tool(
    name="parse_payslip",
    func=parse_payslip,
    description="Parses a payslip PDF file path to extract net monthly salary."
)

# Tool 2: Calculate budget allocations based on salary, loan, and bills


def calculate_budget(net_salary, loan, bills):
    """Calculate budget allocations using 50/30/20 rule, adjusted for debts."""
    disposable = net_salary - loan - bills
    if disposable < 0:
        return {"error": "Disposable income is negative. Adjust debts."}

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
        "Savings": disposable * 0.2
    }
    allocations["Needs Breakdown"] = needs_breakdown
    allocations["Daily"] = {k: v / 30 for k,
                            v in allocations.items() if k != "Needs Breakdown"}
    allocations["Weekly"] = {k: v / 4 for k,
                             v in allocations.items() if k != "Needs Breakdown"}
    return allocations


budget_tool = Tool(
    name="calculate_budget",
    func=lambda inputs: calculate_budget(
        float(inputs["net_salary"]), float(inputs["loan"]), float(inputs["bills"])),
    description="Calculates budget allocations. Inputs: net_salary (float), loan (float), bills (float)."
)

# Tool 3: Generate visualizations (pie and bar charts)


def generate_graphs(allocations):
    """Generate pie chart for monthly allocations and bar charts for slabs."""
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
        # Assume no debts for slabs
        alloc = calculate_budget(s, loan=0, bills=0)
        slab_allocs.append({
            "Salary": s,
            "Needs": alloc["Needs"],
            "Wants": alloc["Wants"],
            "Savings": alloc["Savings"]
        })
    df_slabs = pd.DataFrame(slab_allocs).set_index("Salary")
    fig4, ax4 = plt.subplots()
    df_slabs.plot(kind="bar", stacked=True, ax=ax4)
    ax4.set_title("Budget Slabs from 80k to 5L")
    ax4.set_ylabel("Amount (INR)")
    st.pyplot(fig4)

    return "Graphs generated."


graph_tool = Tool(
    name="generate_graphs",
    func=generate_graphs,
    description="Generates budget graphs based on allocations dict."
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

if st.button("Generate Budget Plan and Dashboard"):
    if uploaded_file:
        # Save uploaded file temporarily
        with open("temp_payslip.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Run agent with prompt
        prompt = f"""
        Parse the payslip from file 'temp_payslip.pdf' to get net salary.
        Then, calculate budget with net_salary, loan={loan_amount}, bills={credit_bills}.
        Finally, generate graphs from the allocations.
        """
        response = agent.run(prompt)
        st.write("Agent Response:", response)

        # Clean up temp file
        os.remove("temp_payslip.pdf")
    else:
        st.error("Please upload a payslip PDF.")
