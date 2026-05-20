"""
Streamlit UI for the Wine Quality classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "churn-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list[float]) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.title("Churn Classifier")
st.write("Enter the measurements below to predict the Churn of users via SageMaker.")

# 1. Setup User Inputs
age=st.number_input("age", 0, 100)
gender=st.radio("gender", ["Male","Female"])
tenure=st.number_input("the period of time you holds a position (in years)", 0,100)
usage_freq=st.number_input("the frequency of product usage (in years)", 0,100)
support_call=st.number_input("number of support calls", 0,10)
payment_delay=st.number_input("the period of payment delay (in months)", 0,30)
subs_type=st.radio("choose subscription type", ["Standard","Premium","Basic"])
contract_length=st.radio("choose contract length", ["Annual","Quarterly","Monthly"])
total_spend=st.number_input("total spend in a month", 0,1000000000)
last_interaction=st.number_input("last interaction with the product (in months)", 0,30)

if st.button("Predict", type="primary"):
    features = [
        Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction, Subscription Type Ordinal,Contract Length Ordinal]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = "Churn" if result == 1 else "Not Churn"
        probs = result["probabilities"][0]

        st.success(f"Predicted quality: **{label}**")
        st.write("Class probabilities:")
        st.bar_chart({"probability": probs})





