import streamlit as st
from azure.storage.blob import BlobClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import re
import json
from azure.search.documents.models import VectorizedQuery
from azure.ai.formrecognizer import DocumentAnalysisClient
import pandas as pd  # Import pandas for handling Excel files
import requests
from dotenv import load_dotenv  # Import dotenv for loading environment variables
import os  # Import os for accessing environment variables
 
# Azure Blob Storage setup
container_url = os.getenv("AZURE_STORAGE_CONTAINER_URL")
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")

# Azure Search setup
# Define the service endpoint and key.
endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
api_key = os.getenv("AZURE_SEARCH_API_KEY")
 
# Initialize the search client
search_client = SearchClient(endpoint=endpoint,
                             index_name="resumes",
                             credential=AzureKeyCredential(api_key))
 
# OpenAI API setup
# Initialize the OpenAI client
client = AzureOpenAI(
  api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_version="2024-02-01"
)
 
# Streamlit app
st.set_page_config(page_title="Resume Ranking for Recruiters", page_icon=":briefcase:", layout="wide")
 
# Resume rank algorithm
def get_openai_score(prompt):
    response = client.chat.completions.create(
        model="nsdcopenai",
        temperature=0,
        messages=[
             {"role": "system", "content": "You are an AI assistant that evaluates resumes based on given criteria such as experience, education, and skills. Your task is to provide a relevance score for the given resume details based on the provided keywords."},
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.choices[0].message.content.strip()
    # Extract the numerical score from the response text
    score_match = re.search(r'(\d+(\.\d+)?)', response_text)
    if score_match:
        score = float(score_match.group(1))
        return score
    else:
        raise ValueError(f"Could not extract score from OpenAI response: {response_text}")
 
 
def generate_prompt(experience, education, skills, keywords):
    prompt = f"""
    Given the following resume details:
    Experience: {experience}
    Education: {education}
    Skills: {skills}
 
    And the following keywords: {keywords}
 
    Score the relevance of this resume for the given keywords on a scale of 0 to 10.
    """
    return prompt

# Define the function to generate the prompt
def generate_prompt_forjd(resume_text, required_qualification, job_details, keywords):
    prompt = f"""
    I will provide you the resume and job description. For the given resume, you have to provide a score to each job description on a scale of 0 to 10 based on the relevance of the resume to the job description.

    Resume:
    {resume_text}

    Job Description:
    Required Qualification: {required_qualification}
    Job Details: {job_details}
    Keywords: {keywords}

    Score the relevance of this resume for the given job description on a scale of 0 to 10.
    """
    return prompt
 
def compute_composite_score(resume, given_keywords):
    experience = " ".join([exp["JobTitle"] for exp in resume.get("Experience", [])])
    education = " ".join([edu["Degree"] for edu in resume.get("Education", [])])
    skills = ", ".join(resume.get("Skills", []))
    experience_in_years = float(resume.get("Experience_in_year", 0))
 
    # Generate the prompt for OpenAI
    prompt = generate_prompt(experience, education, skills, given_keywords)
 
    # Get the score from OpenAI
    openai_score = get_openai_score(prompt)
 
    # Compute job role score and keyword score
    job_roles = resume.get("possibleJobRoles", [])
    keywords_and_scores = resume.get("KeywordsAndScores", [])
    job_role_score = sum(role["Score"] for role in job_roles)
    keyword_score = sum(kw["Score"] for kw in keywords_and_scores)
 
    # Normalize scores
    normalized_job_role_score = job_role_score / (10 * len(job_roles)) if job_roles else 0
    normalized_keyword_score = keyword_score / (10 * len(keywords_and_scores)) if keywords_and_scores else 0
 
    # Combine job role and keyword scores into skills score
    skills_score = (normalized_job_role_score + normalized_keyword_score) / 2
 
    # Calculate individual component scores (out of 100)
    skills_score_percentage = min(skills_score * 100 * 0.3, 30)  # Cap at 30%
    experience_score_percentage = min((experience_in_years / 10) * 100 * 0.4, 40)  # Cap at 40%
    openai_score_percentage = min(openai_score * 10 * 0.3, 30)  # Cap at 30%
 
    # Combine the component scores to get the composite score out of 100
    composite_score = skills_score_percentage + experience_score_percentage + openai_score_percentage
 
    return {
        "composite_score": composite_score,
        "skills_score": skills_score_percentage,
        "experience_score": experience_score_percentage,
        "education_score": openai_score_percentage  # Treated as education_score in this case
    }
 
def suggest_best_resumes(resumes, given_keywords):
    # Compute composite scores for each resume
    for resume in resumes:
        resume["composite_score_details"] = compute_composite_score(resume, given_keywords)
 
    # Sort resumes by composite score in descending order
    sorted_resumes = sorted(resumes, key=lambda x: x["composite_score_details"]["composite_score"], reverse=True)
 
    return sorted_resumes
 
# logo_path = "/Users/abhishekjadon/Desktop/NSDC Code/Web app/pngwing.com.png"
# st.image(logo_path, width=100)
 
# Home tab
def home():
    st.title("Resume Ranking for Recruiters")
    st.write("""
        Welcome to the Resume Ranking tool for recruiters. This tool leverages the power of OpenAI and Azure services to help you efficiently manage and rank resumes based on keywords.
 
        **Features:**
        - Upload resumes directly to Azure Blob Storage.
        - Automatically rank resumes based on keyword relevance.
        - Get keyword suggestions as you type to refine your search.
 
        **How it works:**
        - Upload resumes to the Azure Blob Storage in the 'Resume Upload' tab.
        - Use the 'Resume Ranking' tab to rank resumes based on your specified keywords.
        - Leverage AI-driven keyword suggestions to enhance your search criteria.
    """)
 
# Resume upload tab
def upload_resume():
    st.title("Upload Resume")
    st.write("Upload your resume files to Azure Blob Storage.")
 
    uploaded_files = st.file_uploader("Choose resume files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
 
    if uploaded_files:
        for uploaded_file in uploaded_files:
            blob_name = uploaded_file.name
            blob_client = BlobClient.from_blob_url(f"{container_url}/{blob_name}{sas_token}")
            blob_client.upload_blob(uploaded_file, overwrite=True)
            st.success(f"Resume '{uploaded_file.name}' uploaded successfully!")
 
    # Custom CSS to hide the default limit message
    hide_streamlit_style = """
        <style>
        div[data-testid="stFileUploader"] > div:first-child > div:first-child > div:nth-child(2) {
            display: none;
        }
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
 
# Function to display detailed resume information
def display_resume_details(resume):
    st.write(f"**Name:** {resume['Name']}")
    st.write(f"**Email:** {resume.get('Email', 'N/A')}")
    st.write(f"**Contact Details:** {resume.get('ContactDetails', 'N/A')}")
    st.write(f"**Summary:** {resume.get('Summary', 'N/A')}")
    st.write(f"**Experience in years:** {resume['Experience_in_year']}")
    st.write("**Experience:**")
    for exp in resume.get('Experience', []):
        st.write(f"  - **Job Title:** {exp['JobTitle']}")
        st.write(f"    **Company Name:** {exp['CompanyName']}")
        st.write(f"    **Responsibilities:**")
        for resp in exp['Responsibilities']:
            st.write(f"      - {resp}")
 
    st.write("**Education:**")
    for edu in resume.get('Education', []):
        st.write(f"  - **Degree:** {edu['Degree']}")
        st.write(f"    **Institution:** {edu['Institution']}")
        st.write(f"    **Date:** {edu['Date']}")
        st.write(f"    **Percentage:** {edu.get('Percentage', 'N/A')}")
 
    st.write("**Skills:**")
    for skill in resume.get('Skills', []):
        st.write(f"  - {skill}")
 # Resume ranking tab
def rank_resumes():
    st.title("Resume Ranking")
    st.write("Search and rank resumes based on keywords.")
 
    keyword = st.text_input("Enter keyword(s)", key="keyword_input")
    current_keyword = st.session_state.get('current_keyword', keyword)
 
    if keyword:
        # Get keyword suggestions
        suggestions = get_keyword_suggestions(keyword)
        if suggestions:
            st.write("Keyword Suggestions:")
            # Create a container for buttons to be displayed horizontally
            suggestions_row = st.empty()
            # Render buttons horizontally
            for suggested_keyword in suggestions:
                if st.button(suggested_keyword, key=f"suggested_keyword_{suggested_keyword}"):
                    st.session_state['current_keyword'] = suggested_keyword
                    current_keyword = suggested_keyword
 
    # Perform search with the current keyword (either the user input or clicked suggestion)
    if current_keyword:
        results = search_resumes(current_keyword)
        if results:
            st.write(f"Search Results for '{current_keyword}':")
            resume_list = []
            for result in results:
                resume_list.append(result)
 
            best_resumes = suggest_best_resumes(resume_list, current_keyword.split())
 
            # Ensure experience years are numbers
            for resume in best_resumes:
                resume["Experience_in_year"] = float(resume.get("Experience_in_year", 0))
 
            # Get the range of experience years for the filter
            if best_resumes:
                min_experience = min([resume["Experience_in_year"] for resume in best_resumes])
                max_experience = max([resume["Experience_in_year"] for resume in best_resumes])
            else:
                min_experience = 0
                max_experience = 0  # Set defaults if no resumes found with experience
 
            if min_experience == 0 and max_experience == 0:
                for result in best_resumes:
                    with st.expander(f"{result['Name']} (Score: {result['composite_score_details']['composite_score']:.2f})"):
                        st.write(f"**Composite Score:** {result['composite_score_details']['composite_score']:.2f}")
                        skills_score = result['composite_score_details']['skills_score']
                        experience_score = result['composite_score_details']['experience_score']
                        education_score = result['composite_score_details']['education_score']
 
                        st.write("**Score Distribution:**")
                        st.write(f"**Skills Score:** {skills_score:.2f}")
                        st.progress(min(skills_score / 100, 1.0))
 
                        st.write(f"**Experience Score:** {experience_score:.2f}")
                        st.progress(min(experience_score / 100, 1.0))
 
                        st.write(f"**Education Score:** {education_score:.2f}")
                        st.progress(min(education_score / 100, 1.0))
 
                        st.write(f"**Skills:** {', '.join(result['Skills'][:3])}")
                        display_resume_details(result)
            else:
                # Add a slider to filter resumes based on experience
                experience_range = st.slider("Filter by years of experience:", min_value=min_experience, max_value=max_experience, value=(min_experience, max_experience))
 
                # Filter resumes based on the selected experience range
                filtered_resumes = [resume for resume in best_resumes if experience_range[0] <= resume["Experience_in_year"] <= experience_range[1]]
 
                for result in filtered_resumes:
                    with st.expander(f"{result['Name']} (Score: {result['composite_score_details']['composite_score']:.2f})"):
                        st.write(f"**Composite Score:** {result['composite_score_details']['composite_score']:.2f}")
                        skills_score = result['composite_score_details']['skills_score']
                        experience_score = result['composite_score_details']['experience_score']
                        education_score = result['composite_score_details']['education_score']
 
                        st.write("**Score Distribution:**")
                        st.write(f"**Skills Score:** {skills_score:.2f}")
                        st.progress(min(skills_score / 100, 1.0))
 
                        st.write(f"**Experience Score:** {experience_score:.2f}")
                        st.progress(min(experience_score / 100, 1.0))
 
                        st.write(f"**Education Score:** {education_score:.2f}")
                        st.progress(min(education_score / 100, 1.0))
 
                        st.write(f"**Skills:** {', '.join(result['Skills'][:3])}")
                        display_resume_details(result)


 
def upload_jd():
    st.title("Upload Job Descriptions (JD)")
    st.write("Upload your JD files in Excel format.")
 
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        try:
            # Prepare the files to be uploaded
            files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
           
            # Send POST request to backend URL
            url = "https://doc-upload.azurewebsites.net/api/http_trigger?code=JC0twlLWQqTjFkNKJGkpBRmRS0B16Nf1cz-YcgFEAPlHAzFuAsfs4w%3D%3D"
            response = requests.post(url, files=files)
           
            # Check if request was successful
            if response.status_code == 200:
                data = response.text
               
                # Display popup message from backend response
                st.write(data)
               
            else:
                st.error(f"Failed to retrieve data from backend: HTTP {response.status_code}")
                st.write(response.text)  # Display response text for debugging
        except Exception as e:
            st.error(f"Error occurred while processing the uploaded file: {str(e)}")
 
 
def analyze_resume(uploaded_file):
    with open("temp_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
 
    document_analysis_client = DocumentAnalysisClient(
        endpoint="https://ignoudocintelligence.cognitiveservices.azure.com/",
        credential=AzureKeyCredential("f5f5ee8dd0024821a955fe2d6be2ed1d")
    )
 
    with open("temp_file", "rb") as f:
        poller = document_analysis_client.begin_analyze_document("prebuilt-layout", document=f)
        result = poller.result()
        resume_text = result.content
 
    embedding = generate_embeddings(resume_text)
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=100,
        fields="JobDetailsNoHtmlVector, KeywordsVector, JobIndustryTypeVector, TradeNameVector, TradeSpecializationVector"
    )
 
    endpoint = "https://nsdcaisearch.search.windows.net"
    index_name = "jobindex"
    credential = AzureKeyCredential("VOyABXpQChsSkVZwRDEjAXSuhfd68taBO2JCVARBVKAzSeBiEAKl")
    # Initialize the SearchClient
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
 
    # Perform the vector search
    # results = search_client.search(
    # search_text=None,
    # vector_queries=[vector_query],
    # select=["JobDetails", "Location", "RequiredQualification", "ContactPerson", "JobTitle", "Gender", "Keywords"],
    # include_total_count=True,
    # top=10
    # )
 
    # results = search_client.search(
    #     query_type='semantic',
    #     semantic_configuration_name='my-semantic-config',
    #     # search_text=resume_text,
    #     # search_fields=["JobDetailsNotHtml", "jobExperience1", "Keywords", "RequiredQualification", "TradeName", "Gender"],
    #     # vector_queries=[vector_query],
    #     select=["JobDetails", "Location", "RequiredQualification", "ContactPerson", "JobTitle", "Gender", "Keywords"],
    #     include_total_count=True,
    #     top=10
    # )
 
    results = search_client.search(
        query_type="simple",
        search_text=resume_text,
        select=["JobDetails", "Location", "RequiredQualification", "ContactPerson", "JobTitle", "Gender", "Keywords"],
        search_fields=["JobDetailsNotHtml", "jobExperience1", "Keywords", "RequiredQualification", "TradeName", "Gender"],
        include_total_count=True,
        # filter=filter_criteria,
        # order_by="OverallScore desc",
        top=20
    )
 
    # results = search_client.search(
    #     query_type='semantic',
    #     semantic_configuration_name='my-semantic-config',
    #     search_text=resume_text,
    #     search_fields=["JobDetailsNotHtml", "jobExperience1", "Keywords", "RequiredQualification", "TradeName", "Gender"],
    #     vector_queries=[vector_query],
    #     select=["JobDetails", "Location", "RequiredQualification", "ContactPerson", "JobTitle", "Gender", "Keywords"],
    #     include_total_count=True,
    #     top=5
    # )
    results = list(results)
    print(results)
        # Define a list to store the scored results
    scored_results = []

        # Iterate through each result and generate a score
    for result in results:
        required_qualification = result["RequiredQualification"]
        job_details = result["JobDetails"]
        keywords = result["Keywords"]
        
        prompt = generate_prompt(resume_text, required_qualification, job_details, keywords)
        score = get_openai_score(prompt)
        print()
        
        # Append the result with the score to the list
        scored_results.append((result, score))

    # Sort the results based on the score in descending order
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    for result in scored_results:
        with st.expander(f"Job Title: {result[0]['JobTitle']}"):
            st.write(f"Job Title: {result[0]['JobTitle']}")
            st.write(f"Gender: {result[0]['Gender']}")
            st.write(f"Location: {result[0]['Location']}")
            st.write(f"Required Qualification: {result[0]['RequiredQualification']}")
            st.write(f"Contact Person: {result[0]['ContactPerson']}\n")
            st.markdown(f"Job Description: {result[0]['JobDetails']}", unsafe_allow_html=True)
            st.write(f"Keywords: {result[0]['Keywords']}\n")

    # Optionally, you can return the entire list of sorted results for further processing or display
    return scored_results


def display_results(scored_results):
    for index, result_group in enumerate(scored_results):
        for result in result_group:
            # Extract details with fallback to default values if keys are missing
            job_title = result.get('JobTitle', 'N/A')
            score = result.get('@search.score', 0.0)
            location = result.get('Location', 'N/A')
            gender = result.get('Gender', 'N/A')
            keywords = result.get('Keywords', 'N/A')
            job_details = result.get('JobDetails', 'N/A')
            contact_person = result.get('ContactPerson', 'N/A')
            required_qualification = result.get('RequiredQualification', 'N/A')

            # Create an expandable section for each job listing
            with st.expander(f"Job Title: {job_title} (Score: {score:.2f})"):
                # Display job details
                st.write(f"**Location:** {location}")
                st.write(f"**Gender:** {gender}")
                st.write(f"**Keywords:** {keywords}")
                st.write(f"**Contact Person:** {contact_person}")
                st.write(f"**Required Qualification:** {required_qualification}")
                st.write("**Job Details:**", unsafe_allow_html=True)
                st.markdown(job_details, unsafe_allow_html=True) 


def generate_embeddings(text):
    client = AzureOpenAI(
        api_key="b711f7240d1344c99d39c42f09cc2e50",
        api_version="2024-02-01",
        azure_endpoint="https://nsdcazureopenai.openai.azure.com/",
    )
 
    text_response = client.embeddings.create(input=[text], model="embeddingmodel").data[0].embedding
    return text_response
 
def get_keyword_suggestions(keyword):
    suggester_name = "sg"
    search_text = keyword
    top = 10  # Request more suggestions to ensure enough unique results
 
    # Call the suggester
    suggestions = search_client.suggest(
        search_text=search_text,
        suggester_name=suggester_name,
        top=top,
    )
 
    unique_suggestions = set()
    for suggestion in suggestions:
        text = suggestion['text']
        if text not in unique_suggestions:
            unique_suggestions.add(text)
 
    return unique_suggestions
 
def search_resumes(keyword):
    # search_results = search_client.search(search_text=keyword, include_total_count=True)
    search_results = search_client.search(
        query_type="simple",
        search_text=keyword,
        select="*",
        search_fields=["KeywordsAndScores/Keyword", "possibleJobRoles/Role"],
        include_total_count=True,
    )
    results = list(search_results)
   
    sorted_resumes = suggest_best_resumes(results, keyword)
    return sorted_resumes
 
# Main app
def main():
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Home", "Resume Upload", "Resume Ranking (Keyword based)", "Upload JD", "Resume Analysis"])
 
    if tab == "Home":
        home()
    elif tab == "Resume Upload":
        upload_resume()
    elif tab == "Resume Ranking (Keyword based)":
        rank_resumes()
    elif tab == "Upload JD":
        upload_jd()
    elif tab == "Resume Analysis":
        st.title("Resume Analysis")
        uploaded_file = st.file_uploader("Choose a resume file", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            results = analyze_resume(uploaded_file)
            # print(results)
            # for result in results:
            #     st.write(f"Score: {result['@search.score']}")
            #     st.write(f"Job Title: {result['JobTitle']}")
            #     st.write(f"Gender: {result['Gender']}")
            #     st.write(f"Location: {result['Location']}")
            #     st.write(f"Required Qualification: {result['RequiredQualification']}")
            #     st.write(f"Contact Person: {result['ContactPerson']}\n")
 
 
if __name__ == "__main__":
    main()