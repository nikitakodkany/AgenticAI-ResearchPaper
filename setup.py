from setuptools import setup, find_packages

setup(
    name="research-paper-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.109.2",
        "uvicorn==0.27.1",
        "streamlit==1.31.1",
        "requests==2.31.0",
        "python-dotenv==1.0.1",
        "pydantic==2.6.1",
        "langgraph==0.0.15",
        "langchain==0.1.0",
        "openai==1.12.0",
        "numpy==1.26.4",
        "beautifulsoup4==4.12.3",
        "lxml==5.1.0",
    ],
) 