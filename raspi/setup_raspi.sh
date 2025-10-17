#!/bin/bash
# Raspberry Pi 3B v1.2 Setup Script for Object Detection Testing
# This script sets up Docker and runs the Pi-optimized test suite

set -e

echo "🍓 Raspberry Pi 3B v1.2 Object Detection Setup"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Raspberry Pi
check_pi_hardware() {
    print_status "Checking hardware..."
    
    if [ -f /proc/device-tree/model ]; then
        PI_MODEL=$(cat /proc/device-tree/model | tr -d '\0')
        if [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
            print_success "Detected: $PI_MODEL"
            
            # Check if Pi 3B
            if [[ $PI_MODEL == *"Raspberry Pi 3"* ]]; then
                print_success "Raspberry Pi 3 detected - optimizations will be applied"
            else
                print_warning "Non-Pi 3 detected - some optimizations may not apply"
            fi
        else
            print_warning "Not running on Raspberry Pi hardware"
        fi
    else
        print_warning "Could not detect Pi hardware"
    fi
    
    # Check memory
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
    print_status "Total memory: ${TOTAL_MEM}GB"
    
    if (( $(echo "$TOTAL_MEM < 1.5" | bc -l) )); then
        print_success "Memory suitable for Pi 3B optimization"
    else
        print_warning "Higher memory detected - consider using regular Docker setup"
    fi
}

# Check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_success "Docker found: $DOCKER_VERSION"
    else
        print_error "Docker not found! Installing Docker..."
        
        # Install Docker for Raspberry Pi
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        
        print_success "Docker installed! Please logout and login again, then re-run this script"
        exit 1
    fi
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version)
        print_success "Docker Compose found: $COMPOSE_VERSION"
    else
        print_error "Docker Compose not found! Installing..."
        
        # Install Docker Compose for ARM
        sudo pip3 install docker-compose
        print_success "Docker Compose installed!"
    fi
}

# Build Pi-optimized Docker image
build_pi_image() {
    print_status "Building Raspberry Pi optimized Docker image..."
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found!"
        exit 1
    fi
    
    # Build with Pi optimizations
    print_status "Building ARM64 image (this may take 15-30 minutes on Pi 3B)..."
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        print_success "Pi Docker image built successfully!"
    else
        print_error "Failed to build Pi Docker image"
        exit 1
    fi
}

# Run Pi tests
run_pi_tests() {
    print_status "Running Raspberry Pi test suite..."
    
    # Start the container
    print_status "Starting Pi container..."
    docker-compose up -d
    
    # Wait for container to be ready
    print_status "Waiting for container startup..."
    sleep 10
    
    # Run the Pi-specific tests
    print_status "Executing Pi test suite..."
    docker-compose exec -T raspi-detection python3 /workspace/tests/test_raspi_detection.py
    
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -eq 0 ]; then
        print_success "All Pi tests passed! 🎉"
    else
        print_warning "Some tests failed - check output above"
    fi
    
    return $TEST_RESULT
}

# Run obj_detection.py test
run_obj_detection_test() {
    print_status "Testing obj_detection.py on Pi..."
    
    # Copy current obj_detection.py if it exists
    if [ -f "../scripts/obj_detection.py" ]; then
        print_status "Copying obj_detection.py from main project"
        cp ../scripts/obj_detection.py scripts/
    elif [ -f "scripts/obj_detection.py" ]; then
        print_status "Using existing obj_detection.py"
    else
        print_warning "obj_detection.py not found in scripts/"
        return 1
    fi
    
    # Run the actual detection script
    print_status "Running obj_detection.py in Pi container..."
    docker-compose exec -T raspi-detection bash -c "
        cd /workspace/scripts && 
        python3 -c '
import sys
sys.path.append(\"/workspace/scripts\")
from obj_detection import is_raspberry_pi
is_pi, info = is_raspberry_pi()
print(f\"Pi detection: {is_pi}\")
print(f\"Info: {info}\")
print(\"✅ obj_detection.py working on Pi container!\")
'
    "
    
    if [ $? -eq 0 ]; then
        print_success "obj_detection.py verified on Pi!"
    else
        print_error "obj_detection.py test failed"
        return 1
    fi
}

# Performance benchmark
run_performance_test() {
    print_status "Running Pi performance benchmark..."
    
    docker-compose exec -T raspi-detection python3 -c "
import time
import psutil
from ultralytics import YOLO
import numpy as np

print('🍓 Pi Performance Test')
print('======================')

# System info
memory = psutil.virtual_memory()
print(f'Available RAM: {memory.available / (1024**2):.0f} MB')
print(f'CPU cores: {psutil.cpu_count()}')

# Load model
print('Loading YOLO11n...')
start = time.time()
model = YOLO('yolo11n.pt')
load_time = time.time() - start
print(f'Model load time: {load_time:.2f}s')

# Test inference
test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
print('Running inference test...')

times = []
for i in range(3):
    start = time.time()
    results = model(test_image, verbose=False)
    inference_time = time.time() - start
    times.append(inference_time * 1000)
    print(f'  Run {i+1}: {inference_time*1000:.1f}ms')

avg_time = sum(times) / len(times)
fps = 1000 / avg_time

print(f'Average inference: {avg_time:.1f}ms')
print(f'Estimated FPS: {fps:.1f}')

# Memory usage
final_memory = psutil.virtual_memory()
print(f'Final RAM usage: {100 - final_memory.percent:.1f}% available')

print('✅ Performance test completed!')
"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    docker-compose down
    print_success "Cleanup completed"
}

# Main execution
main() {
    echo ""
    print_status "Starting Raspberry Pi setup process..."
    
    # Check hardware
    check_pi_hardware
    echo ""
    
    # Check Docker
    check_docker
    echo ""
    
    # Build image
    build_pi_image
    echo ""
    
    # Run tests
    print_status "Running comprehensive Pi test suite..."
    run_pi_tests
    echo ""
    
    # Test obj_detection.py
    run_obj_detection_test
    echo ""
    
    # Performance test
    run_performance_test
    echo ""
    
    print_success "🎉 Raspberry Pi setup and testing completed!"
    print_status "Your obj_detection.py is ready to run on Raspberry Pi 3B v1.2"
    
    echo ""
    echo "Quick start commands:"
    echo "  Start container: docker-compose up -d"
    echo "  Run detection:   docker-compose exec raspi-detection python3 /workspace/scripts/obj_detection.py"
    echo "  Stop container:  docker-compose down"
    echo ""
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root - some Docker commands may need adjustment"
fi

# Run main function
main "$@"