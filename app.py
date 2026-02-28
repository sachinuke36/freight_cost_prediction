import streamlit as st
import pandas as pd
import numpy as np
from inference.predict_freight import predict_freight
from inference.predict_invoice_flag import predict_invoice_flag


# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)


# ----------------------------------------------
# Header Section
# ----------------------------------------------
st.markdown("""
# 📦 Vendor Invoice Intelligence Portal

### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to

- **Forecast freight costs accurately**

- **Detect risky or abnormal vendor invoices**

- **Reduce financial leakage and manual workload**

""")

st.divider()

# ------------------------------------------
# Sidebar
# ------------------------------------------
st.sidebar.title("🔎 Model Selection")
selectedModel = st.sidebar.radio(
    "Choose prediction module",
    [
        "Freight cost prediction",
        "Invoice manual approval flag"
    ]
)
st.sidebar.markdown(
    """
    ---
    **Business Impact**
    - 📈 Improved cost forecasting
    - 📉 reduced invoice fraud and anamolies
    - ⚙️ Faster Finance operations
    """
)

# ----------------------------------------------
# Freight cost prediction
# ----------------------------------------------
if selectedModel == 'Freight cost prediction':
    st.subheader('🚚 Freight Cost Prediction')
    st.markdown("""
                **Objective:**
                Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
                to support budgeting, forecasting, and vendor negotiations.
                """)
    
    with st.form("frieght_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            quantity = st.number_input(
                "📦 Quantity",
                min_value=1,
                value = 1200 
            )
        with col2:
            dollars = st.number_input(
                "💰 Invoice dollars",
                min_value=1.0,
                value=18500.0
            )
            
        submit_freight = st.form_submit_button("Predict freight cost")
        
    if submit_freight:
        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }
        prediction = predict_freight(input_data)["Predicted_Freight"]
        
        st.success("Prediction Completed Successfull!")
        st.metric(
            label="Estimated Freight cost",
            value=f"${prediction[0]:.2f}"
        )
# --------------------------------
# Invoice Flag Prediction
# --------------------------------
elif selectedModel == 'Invoice manual approval flag':
    st.subheader('🚩 Invoice Manual Approval Flag Prediction')
    st.markdown("""
                **Objective:**
                Identify potentially suspicious or anomalous vendor invoices that require manual review.
                This helps prevent fraud, overpayments, and billing errors by flagging high-risk transactions.
                """)

    st.info("**Model Input**: Provide invoice details and corresponding purchase order information for risk assessment.")

    with st.form("invoice_flag_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Invoice Details")
            invoice_quantity = st.number_input(
                "📦 Invoice Quantity",
                min_value=1,
                value=1200,
                help="Quantity listed on the vendor invoice"
            )
            invoice_dollars = st.number_input(
                "💵 Invoice Dollars",
                min_value=0.01,
                value=18500.00,
                help="Total dollar amount on the vendor invoice"
            )
            freight = st.number_input(
                "🚚 Freight Cost",
                min_value=0.0,
                value=350.00,
                help="Freight/shipping cost charged on the invoice"
            )

        with col2:
            st.markdown("##### Purchase Order Details")
            total_item_quantity = st.number_input(
                "📋 PO Total Quantity",
                min_value=1,
                value=1200,
                help="Total quantity from the corresponding purchase order"
            )
            total_item_dollars = st.number_input(
                "💰 PO Total Dollars",
                min_value=0.01,
                value=18500.00,
                help="Total dollar amount from the corresponding purchase order"
            )

        submit_flag = st.form_submit_button("Predict Invoice Risk Flag")

    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        prediction_df = predict_invoice_flag(input_data)
        flag = prediction_df["Predicted_Flag"].iloc[0]

        st.success("Risk Assessment Completed Successfully!")

        # Display result with color coding
        if flag == 0:
            st.success("### ✅ NORMAL INVOICE")
            st.markdown("""
                **Risk Level**: Low
                **Recommendation**: This invoice appears normal and can proceed through standard approval workflow.
            """)
        else:
            st.error("### ⚠️ SUSPICIOUS INVOICE - MANUAL REVIEW REQUIRED")
            st.markdown("""
                **Risk Level**: High
                **Recommendation**: This invoice has been flagged for anomalies. Please conduct manual review before approval.

                **Possible Issues to Check:**
                - Invoice quantities don't match PO quantities
                - Invoice amounts significantly differ from PO amounts
                - Unusual freight charges
                - Potential duplicate or fraudulent invoice
            """)

        # Display input details in expandable section
        with st.expander("View Input Details"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Invoice Information**")
                st.write(f"Quantity: {invoice_quantity:,}")
                st.write(f"Dollars: ${invoice_dollars:,.2f}")
                st.write(f"Freight: ${freight:,.2f}")
            with col_b:
                st.markdown("**Purchase Order Information**")
                st.write(f"PO Quantity: {total_item_quantity:,}")
                st.write(f"PO Dollars: ${total_item_dollars:,.2f}")


# ----------------------------------------------
# Footer
# ----------------------------------------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🤖 Powered by Machine Learning | Built for Finance Operations Excellence</p>
        <p style='font-size: 0.85em;'>This tool is designed to augment human decision-making, not replace it.</p>
    </div>
""", unsafe_allow_html=True)