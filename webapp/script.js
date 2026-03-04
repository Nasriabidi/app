/**
 * Crack Inspection AI - Frontend Application
 */

const API_URL = 'http://3.80.179.127:8000';

// DOM Elements
const elements = {
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadPreview: document.getElementById('uploadPreview'),
    previewImage: document.getElementById('previewImage'),
    clearUpload: document.getElementById('clearUpload'),
    analyzeUpload: document.getElementById('analyzeUpload'),
    
    // Camera
    cameraPreview: document.getElementById('cameraPreview'),
    cameraCanvas: document.getElementById('cameraCanvas'),
    cameraPlaceholder: document.getElementById('cameraPlaceholder'),
    startCamera: document.getElementById('startCamera'),
    captureAnalyze: document.getElementById('captureAnalyze'),
    
    // Result
    resultCard: document.getElementById('resultCard'),
    resultImage: document.getElementById('resultImage'),
    downloadResult: document.getElementById('downloadResult'),
    
    // Loading & Error
    loadingOverlay: document.getElementById('loadingOverlay'),
    errorToast: document.getElementById('errorToast'),
    errorMessage: document.getElementById('errorMessage'),
};

// State
let selectedFile = null;
let cameraStream = null;
let isProcessing = false;

// Initialize
document.addEventListener('DOMContentLoaded', init);

function init() {
    setupUploadHandlers();
    setupCameraHandlers();
    setupResultHandlers();
}

// ==================
// Upload Handlers
// ==================

function setupUploadHandlers() {
    // Click to upload
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });

    // File selection
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });

    // Clear upload
    elements.clearUpload.addEventListener('click', clearUpload);

    // Analyze button
    elements.analyzeUpload.addEventListener('click', analyzeUploadedImage);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.uploadArea.hidden = true;
        elements.uploadPreview.hidden = false;
        elements.analyzeUpload.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    selectedFile = null;
    elements.fileInput.value = '';
    elements.previewImage.src = '';
    elements.uploadArea.hidden = false;
    elements.uploadPreview.hidden = true;
    elements.analyzeUpload.disabled = true;
}

async function analyzeUploadedImage() {
    if (!selectedFile || isProcessing) return;
    await sendImageToAPI(selectedFile);
}

// ==================
// Camera Handlers
// ==================

function setupCameraHandlers() {
    elements.startCamera.addEventListener('click', toggleCamera);
    elements.captureAnalyze.addEventListener('click', captureAndAnalyze);
}

async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        // Request camera with preference for back camera on mobile
        const constraints = {
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.cameraPreview.srcObject = cameraStream;
        elements.cameraPlaceholder.hidden = true;
        elements.captureAnalyze.disabled = false;
        
        // Update button text
        elements.startCamera.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
            </svg>
            Stop Camera
        `;
    } catch (error) {
        console.error('Camera error:', error);
        showError('Unable to access camera. Please check permissions.');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    elements.cameraPreview.srcObject = null;
    elements.cameraPlaceholder.hidden = false;
    elements.captureAnalyze.disabled = true;
    
    // Update button text
    elements.startCamera.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Start Camera
    `;
}

async function captureAndAnalyze() {
    if (!cameraStream || isProcessing) return;

    const video = elements.cameraPreview;
    const canvas = elements.cameraCanvas;
    
    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert to blob
    canvas.toBlob(async (blob) => {
        if (blob) {
            const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            await sendImageToAPI(file);
        }
    }, 'image/jpeg', 0.9);
}

// ==================
// API Integration
// ==================

async function sendImageToAPI(file) {
    if (isProcessing) return;
    
    isProcessing = true;
    showLoading(true);
    disableButtons(true);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `Server error: ${response.status}`);
        }
        
        // Get the processed image
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        // Display result
        displayResult(imageUrl);
        
    } catch (error) {
        console.error('API error:', error);
        
        if (error.message.includes('Failed to fetch')) {
            showError('Cannot connect to server. Make sure the API is running.');
        } else {
            showError(error.message || 'An error occurred during analysis');
        }
    } finally {
        isProcessing = false;
        showLoading(false);
        disableButtons(false);
    }
}

// ==================
// Result Handlers
// ==================

function setupResultHandlers() {
    elements.downloadResult.addEventListener('click', downloadResult);
}

function displayResult(imageUrl) {
    elements.resultImage.src = imageUrl;
    elements.resultCard.hidden = false;
    
    // Scroll to result
    elements.resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadResult() {
    const imageUrl = elements.resultImage.src;
    if (!imageUrl) return;
    
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `crack-detection-${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ==================
// UI Helpers
// ==================

function showLoading(show) {
    elements.loadingOverlay.hidden = !show;
}

function disableButtons(disabled) {
    elements.analyzeUpload.disabled = disabled || !selectedFile;
    elements.captureAnalyze.disabled = disabled || !cameraStream;
    elements.startCamera.disabled = disabled;
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorToast.hidden = false;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        elements.errorToast.hidden = true;
    }, 5000);
}

// ==================
// Cleanup
// ==================

window.addEventListener('beforeunload', () => {
    if (cameraStream) {
        stopCamera();
    }
});
