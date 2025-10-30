# 1. Start from a Conda base image
FROM continuumio/miniconda3

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the environment file FIRST
COPY environment.yml .

# 4. Create the Conda environment from the file
# This is the slowest step, so we do it first to cache it
RUN conda env create -f environment.yml

# 5. Clean up Conda cache
RUN conda clean -a -y

# 6. Set the default shell to use Conda
SHELL ["/bin/bash", "-c"]

# 7. Make all future commands run inside the 'cvmatching' environment
# This is the correct way to "activate" the env for Docker
ENV PATH /opt/conda/envs/cvmatching/bin:$PATH

# 8. Copy the rest of your project code into the app directory
COPY . .

# 9. Expose the default Streamlit port
EXPOSE 8501

# 10. The command to run when the container starts
# This runs `streamlit run app.py` from within the activated env
CMD ["streamlit", "run", "app.py"]
