/**
 * Crack Inspection AI - Frontend Application
 * Enhanced Version with Modern UX
 */

// API URL - Uses nginx proxy in Docker, falls back to direct URL for development
const API_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "/api" // Docker: nginx proxies /api/* to api container
    : `${window.location.protocol}//${window.location.host}/api`; // Production

// DOM Elements
const elements = {
  // Upload
  uploadArea: document.getElementById("uploadArea"),
  fileInput: document.getElementById("fileInput"),
  uploadPreview: document.getElementById("uploadPreview"),
  previewImage: document.getElementById("previewImage"),
  clearUpload: document.getElementById("clearUpload"),
  analyzeUpload: document.getElementById("analyzeUpload"),

  // Camera
  cameraPreview: document.getElementById("cameraPreview"),
  cameraCanvas: document.getElementById("cameraCanvas"),
  cameraPlaceholder: document.getElementById("cameraPlaceholder"),
  startCamera: document.getElementById("startCamera"),
  captureAnalyze: document.getElementById("captureAnalyze"),

  // Result
  resultCard: document.getElementById("resultCard"),
  resultImage: document.getElementById("resultImage"),
  downloadResult: document.getElementById("downloadResult"),
  newAnalysis: document.getElementById("newAnalysis"),

  // Loading & Error
  loadingOverlay: document.getElementById("loadingOverlay"),
  errorToast: document.getElementById("errorToast"),
  errorMessage: document.getElementById("errorMessage"),
  closeToast: document.getElementById("closeToast"),
};

// State
let selectedFile = null;
let cameraStream = null;
let isProcessing = false;

// Initialize
document.addEventListener("DOMContentLoaded", init);

function init() {
  setupUploadHandlers();
  setupCameraHandlers();
  setupResultHandlers();
  setupToastHandlers();
}

// ==================
// Upload Handlers
// ==================

function setupUploadHandlers() {
  // Click to upload
  elements.uploadArea.addEventListener("click", () => {
    elements.fileInput.click();
  });

  // File selection
  elements.fileInput.addEventListener("change", handleFileSelect);

  // Drag and drop
  elements.uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.uploadArea.classList.add("dragover");
  });

  elements.uploadArea.addEventListener("dragleave", (e) => {
    e.preventDefault();
    elements.uploadArea.classList.remove("dragover");
  });

  elements.uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.uploadArea.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith("image/")) {
      handleFile(files[0]);
    } else {
      showError("Please drop a valid image file");
    }
  });

  // Clear upload
  elements.clearUpload.addEventListener("click", (e) => {
    e.stopPropagation();
    clearUpload();
  });

  // Analyze button
  elements.analyzeUpload.addEventListener("click", analyzeUploadedImage);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleFile(file);
  }
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select a valid image file");
    return;
  }

  // Check file size (max 10MB)
  if (file.size > 10 * 1024 * 1024) {
    showError("File size must be less than 10MB");
    return;
  }

  selectedFile = file;

  // Show preview with animation
  const reader = new FileReader();
  reader.onload = (e) => {
    elements.previewImage.src = e.target.result;
    elements.uploadArea.hidden = true;
    elements.uploadPreview.hidden = false;
    elements.analyzeUpload.disabled = false;

    // Add entrance animation
    elements.uploadPreview.style.animation = "none";
    elements.uploadPreview.offsetHeight; // Trigger reflow
    elements.uploadPreview.style.animation = "fadeIn 0.3s ease";
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  selectedFile = null;
  elements.fileInput.value = "";
  elements.previewImage.src = "";
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
  elements.startCamera.addEventListener("click", toggleCamera);
  elements.captureAnalyze.addEventListener("click", captureAndAnalyze);
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
        facingMode: {ideal: "environment"},
        width: {ideal: 1920},
        height: {ideal: 1080},
      },
    };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    elements.cameraPreview.srcObject = cameraStream;
    elements.cameraPlaceholder.hidden = true;
    elements.captureAnalyze.disabled = false;

    // Update button
    updateCameraButton(true);
  } catch (error) {
    console.error("Camera error:", error);

    if (error.name === "NotAllowedError") {
      showError("Camera access denied. Please allow camera permissions.");
    } else if (error.name === "NotFoundError") {
      showError("No camera found on this device.");
    } else {
      showError("Unable to access camera. Please check permissions.");
    }
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  elements.cameraPreview.srcObject = null;
  elements.cameraPlaceholder.hidden = false;
  elements.captureAnalyze.disabled = true;

  // Update button
  updateCameraButton(false);
}

function updateCameraButton(isActive) {
  if (isActive) {
    elements.startCamera.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
            </svg>
            <span>Stop Camera</span>
        `;
    elements.startCamera.classList.add("camera-active");
  } else {
    elements.startCamera.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            <span>Start Camera</span>
        `;
    elements.startCamera.classList.remove("camera-active");
  }
}

async function captureAndAnalyze() {
  if (!cameraStream || isProcessing) return;

  const video = elements.cameraPreview;
  const canvas = elements.cameraCanvas;

  // Set canvas size to video size
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw video frame to canvas
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  // Add capture flash effect
  addCaptureFlash();

  // Convert to blob
  canvas.toBlob(
    async (blob) => {
      if (blob) {
        const file = new File([blob], `capture_${Date.now()}.jpg`, {type: "image/jpeg"});
        await sendImageToAPI(file);
      }
    },
    "image/jpeg",
    0.92,
  );
}

function addCaptureFlash() {
  const flash = document.createElement("div");
  flash.style.cssText = `
        position: fixed;
        inset: 0;
        background: white;
        z-index: 9999;
        animation: flash 0.3s ease-out forwards;
    `;
  document.body.appendChild(flash);

  // Add animation keyframes dynamically
  if (!document.getElementById("flash-style")) {
    const style = document.createElement("style");
    style.id = "flash-style";
    style.textContent = `
            @keyframes flash {
                0% { opacity: 0.8; }
                100% { opacity: 0; }
            }
        `;
    document.head.appendChild(style);
  }

  setTimeout(() => flash.remove(), 300);
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
    formData.append("file", file);

    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errorText;
      try {
        errorText = await response.text();
      } catch {
        errorText = `Server error: ${response.status}`;
      }
      throw new Error(errorText || `Server error: ${response.status}`);
    }

    // Get the processed image
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);

    // Display result
    displayResult(imageUrl);
  } catch (error) {
    console.error("API error:", error);

    if (error.name === "TypeError" && error.message.includes("Failed to fetch")) {
      showError("Cannot connect to server. Make sure the API is running.");
    } else if (error.message.includes("NetworkError")) {
      showError("Network error. Please check your internet connection.");
    } else {
      showError(error.message || "An error occurred during analysis");
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
  elements.downloadResult.addEventListener("click", downloadResult);
  elements.newAnalysis.addEventListener("click", resetForNewAnalysis);
}

function displayResult(imageUrl) {
  elements.resultImage.src = imageUrl;
  elements.resultCard.hidden = false;

  // Scroll to result smoothly
  setTimeout(() => {
    elements.resultCard.scrollIntoView({behavior: "smooth", block: "center"});
  }, 100);
}

function downloadResult() {
  const imageUrl = elements.resultImage.src;
  if (!imageUrl) return;

  const link = document.createElement("a");
  link.href = imageUrl;
  link.download = `crack-detection-${Date.now()}.jpg`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function resetForNewAnalysis() {
  // Hide result card with animation
  elements.resultCard.style.animation = "fadeOut 0.3s ease forwards";

  setTimeout(() => {
    elements.resultCard.hidden = true;
    elements.resultCard.style.animation = "";

    // Clear uploads
    clearUpload();

    // Scroll to top
    window.scrollTo({top: 0, behavior: "smooth"});
  }, 300);

  // Add fadeOut animation if not exists
  if (!document.getElementById("fadeout-style")) {
    const style = document.createElement("style");
    style.id = "fadeout-style";
    style.textContent = `
            @keyframes fadeOut {
                from { opacity: 1; transform: translateY(0); }
                to { opacity: 0; transform: translateY(20px); }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }
        `;
    document.head.appendChild(style);
  }
}

// ==================
// Toast Handlers
// ==================

function setupToastHandlers() {
  if (elements.closeToast) {
    elements.closeToast.addEventListener("click", hideError);
  }
}

// ==================
// UI Helpers
// ==================

function showLoading(show) {
  elements.loadingOverlay.hidden = !show;

  // Prevent body scroll when loading
  document.body.style.overflow = show ? "hidden" : "";
}

function disableButtons(disabled) {
  elements.analyzeUpload.disabled = disabled || !selectedFile;
  elements.captureAnalyze.disabled = disabled || !cameraStream;
  elements.startCamera.disabled = disabled;
}

let errorTimeout;

function showError(message) {
  // Clear existing timeout
  if (errorTimeout) {
    clearTimeout(errorTimeout);
  }

  elements.errorMessage.textContent = message;
  elements.errorToast.hidden = false;

  // Auto-hide after 6 seconds
  errorTimeout = setTimeout(() => {
    hideError();
  }, 6000);
}

function hideError() {
  elements.errorToast.hidden = true;
  if (errorTimeout) {
    clearTimeout(errorTimeout);
    errorTimeout = null;
  }
}

// ==================
// Cleanup
// ==================

window.addEventListener("beforeunload", () => {
  if (cameraStream) {
    stopCamera();
  }
});

// Handle visibility change (stop camera when tab is hidden)
document.addEventListener("visibilitychange", () => {
  if (document.hidden && cameraStream) {
    // Optionally stop camera when tab is not visible
    // stopCamera();
  }
});
