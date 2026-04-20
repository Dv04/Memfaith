FROM continuumio/miniconda3

WORKDIR /app

# Install system dependencies if required by vllm/pytorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment spec
COPY environment.yml .

# Create the conda environment and clean cache
RUN conda env create -f environment.yml && conda clean -afy

# Setup default shell to automatically activate the environment inside the RUN commands
SHELL ["conda", "run", "-n", "memfaith", "/bin/bash", "-c"]

# Download necessary heuristical dependencies
RUN python -m spacy download en_core_web_sm

# Copy the remaining project codebase
COPY . .

# Ensure bash scripts have execution permissions
RUN chmod +x run_all.sh setup_env.sh

# Set the entrypoint to the conda execution wrapper
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "memfaith"]

# Default command if no arguments are passed
CMD ["./run_all.sh"]
