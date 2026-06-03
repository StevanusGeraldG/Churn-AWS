"""
Streamlit UI for the Churn classifier hosted on SageMaker.

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


def invoke_endpoint(features: list) -> dict:
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
age = st.number_input("Age", 0, 100, value=25)
gender = st.radio("Gender", ["Male", "Female"])
tenure = st.number_input("The period of time you hold a position (in years)", 0, 100, value=1)
usage_freq = st.number_input("The frequency of product usage (in times/month)", 0, 100, value=5)
support_call = st.number_input("Number of support calls", 0, 10, value=0)
payment_delay = st.number_input("The period of payment delay (in months)", 0, 30, value=0)
subs_type = st.radio("Choose subscription type", ["Basic", "Standard", "Premium"])
contract_length = st.radio("Choose contract length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total spend in a month", 0, 1000000000, value=100000)
last_interaction = st.number_input("Last interaction with the product (in days)", 0, 30, value=2)

if st.button("Predict", type="primary"):
    # 2. Proses Mapping/Encoding Teks ke Angka (Sesuaikan dengan urutan pas training modelmu dulu)
    gender_encoded = 1 if gender == "Male" else 0
    
    # Contoh mapping Ordinal untuk Subscription Type
    subs_mapping = {"Basic": 0, "Standard": 1, "Premium": 2}
    subs_encoded = subs_mapping[subs_type]
    
    # Contoh mapping Ordinal untuk Contract Length
    contract_mapping = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    contract_encoded = contract_mapping[contract_length]

    # 3. Susun sesuai urutan fitur yang diminta oleh Model SageMaker kamu!
    # Pastikan urutan array ini SAMA PERSIS dengan urutan kolom saat kamu training model.
    features = [
        float(age), 
        float(gender_encoded), 
        float(tenure), 
        float(usage_freq), 
        float(support_call), 
        float(payment_delay), 
        float(total_spend), 
        float(last_interaction), 
        float(subs_encoded), 
        float(contract_encoded)
    ]
    
    try:
        result = invoke_endpoint(features)
        
        # Tergantung output model SageMaker (bisa berupa angka langsung atau dictionary)
        # Di bawah ini asumsi jika outputnya berupa json standar list/prediksi langsung
        if isinstance(result, list):
            prediction = result[0]
        else:
            # Jika mengembalikan dictionary seperti struktur kodemu sebelumnya
            prediction = result.get("predictions", [result])[0]
            
        # Menentukan Label (Sesuaikan logika index output modelmu)
        label = "Churn" if prediction == 1 else "Not Churn"
        st.success(f"Prediction Result: **{label}**")
        
        # Bagian probability dicomment dulu jika struktur output model dari SageMaker belum fiks
        # probs = result["probabilities"][0]
        # st.write("Class probabilities:")
        # st.bar_chart({"probability": probs})

    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
