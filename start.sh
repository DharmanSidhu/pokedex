#!/bin/bash
# ============================================================================
# AI Pokedex — One-command launcher
# Starts the MLX LLM server and Streamlit app together
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="mlx-community/Qwen3.5-9B-MLX-4bit"
MLX_PORT=8080
STREAMLIT_PORT=8501
MLX_PID=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [ -n "$MLX_PID" ] && kill -0 "$MLX_PID" 2>/dev/null; then
        echo -e "${CYAN}Stopping MLX server (PID: $MLX_PID)...${NC}"
        kill "$MLX_PID" 2>/dev/null
        wait "$MLX_PID" 2>/dev/null
    fi
    echo -e "${GREEN}AI Pokedex shut down cleanly.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# --- Pre-flight checks ---
echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       AI POKEDEX — Startup           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found. Install Python 3.11+.${NC}"
    exit 1
fi

# Check if an active LLM server is already running (e.g. LM Studio on port 1234 or MLX on port 8080)
ALREADY_RUNNING=0
ACTIVE_PORT=0

for port in 1234 8080; do
    if curl -s "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
        ALREADY_RUNNING=1
        ACTIVE_PORT=$port
        break
    fi
done

if [ $ALREADY_RUNNING -eq 1 ]; then
    echo -e "${GREEN}  ✓ Active local LLM server detected running on port $ACTIVE_PORT!${NC}"
    echo -e "${GREEN}  Skipping model checks and local server startup.${NC}"
else
    # Check if model is downloaded
    echo -e "${CYAN}[1/4] Checking model availability...${NC}"
    if python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_files_only=True)" 2>/dev/null; then
        echo -e "${GREEN}  ✓ Model found locally${NC}"
    else
        echo -e "${YELLOW}  ⚠ Model not found locally. Downloading (≈5.6 GB)...${NC}"
        echo -e "${YELLOW}  This is a one-time download.${NC}"
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"
        echo -e "${GREEN}  ✓ Model downloaded${NC}"
    fi
fi

# Check if data pipeline has been run
echo -e "${CYAN}[2/4] Checking knowledge base...${NC}"
if [ -d "./chroma_pokemon_kb" ] && [ "$(ls -A ./chroma_pokemon_kb 2>/dev/null)" ]; then
    echo -e "${GREEN}  ✓ Pokemon knowledge base found${NC}"
else
    echo -e "${YELLOW}  ⚠ Knowledge base not found. Running data pipeline...${NC}"
    .venv/bin/python3 -m data_pipeline.etl
    echo -e "${GREEN}  ✓ Knowledge base built${NC}"
fi

if [ $ALREADY_RUNNING -eq 0 ]; then
    # Start MLX server
    echo -e "${CYAN}[3/4] Starting MLX LLM server on port $MLX_PORT...${NC}"
    python3 -m mlx_lm.server \
        --model "$MODEL" \
        --port "$MLX_PORT" \
        --host 127.0.0.1 &
    MLX_PID=$!

    # Wait for MLX server to be ready
    echo -e "${CYAN}  Waiting for MLX server to initialize...${NC}"
    MAX_WAIT=120
    WAITED=0
    while ! curl -s "http://127.0.0.1:$MLX_PORT/v1/models" > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo -e "${RED}  ✗ MLX server failed to start within ${MAX_WAIT}s${NC}"
            exit 1
        fi
        echo -e "${CYAN}  ... waiting ($WAITED s)${NC}"
    done
    echo -e "${GREEN}  ✓ MLX server ready${NC}"
else
    echo -e "${GREEN}[3/4] Connecting to already active server on port $ACTIVE_PORT... Ready${NC}"
fi


# Start Streamlit
echo -e "${CYAN}[4/4] Starting Streamlit UI on port $STREAMLIT_PORT...${NC}"
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  AI Pokedex is ready!                ║${NC}"
echo -e "${GREEN}║  Open: http://localhost:$STREAMLIT_PORT        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"

.venv/bin/streamlit run app/main.py \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false

