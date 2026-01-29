// Global variables
let selectedFile = null;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop support
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }
});

// File selection handler
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && isValidImageFile(file)) {
        selectedFile = file;
        displayImagePreview(file);
    } else {
        showError('Please select a valid image file (PNG, JPG, or JPEG)');
    }
}

// Validate image file
function isValidImageFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    return validTypes.includes(file.type);
}

// Display image preview
function displayImagePreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        hideAllSections();
    };
    
    reader.readAsDataURL(file);
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#007bff';
    uploadArea.style.backgroundColor = '#f0f8ff';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#dee2e6';
    uploadArea.style.backgroundColor = 'transparent';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file && isValidImageFile(file)) {
        selectedFile = file;
        displayImagePreview(file);
    } else {
        showError('Please drop a valid image file (PNG, JPG, or JPEG)');
    }
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) {
        showError('No image selected');
        return;
    }
    
    // Show loading
    hideAllSections();
    loadingSection.style.display = 'block';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        // Send to server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to analyze image. Please try again.');
    } finally {
        loadingSection.style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    hideAllSections();
    
    // Update result image
    document.getElementById('resultImage').src = `/static/uploads/${data.filename}`;
    
    // Update disease name
    document.getElementById('diseaseName').textContent = data.predicted_class;
    
    // Update confidence badge
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.textContent = data.risk_text;
    confidenceBadge.className = `confidence-badge ${data.risk_level}`;
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceFill.style.width = confidencePercent + '%';
    
    // Set bar color based on risk level
    if (data.risk_level === 'high') {
        confidenceFill.style.backgroundColor = '#28a745';
    } else if (data.risk_level === 'medium') {
        confidenceFill.style.backgroundColor = '#ffc107';
    } else {
        confidenceFill.style.backgroundColor = '#dc3545';
    }
    
    // Update confidence text
    document.getElementById('confidenceText').textContent = 
        `Confidence: ${confidencePercent}% | ${data.timestamp}`;
    
    // Display all predictions
    displayAllPredictions(data.all_predictions);
    
    // Show results section
    resultsSection.style.display = 'block';
}

// Display all predictions
function displayAllPredictions(predictions) {
    const container = document.getElementById('allPredictions');
    container.innerHTML = '';
    
    // Sort predictions by value (descending)
    const sortedPredictions = Object.entries(predictions)
        .sort((a, b) => b[1] - a[1]);
    
    sortedPredictions.forEach(([disease, probability]) => {
        const percent = Math.round(probability * 100);
        
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span>${disease}</span>
            <div class="prediction-bar">
                <div class="prediction-bar-fill" style="width: ${percent}%"></div>
            </div>
            <strong>${percent}%</strong>
        `;
        
        container.appendChild(item);
    });
}

// Show error
function showError(message) {
    hideAllSections();
    document.getElementById('errorMessage').textContent = message;
    errorSection.style.display = 'block';
}

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    
    hideAllSections();
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
}

// Hide all sections
function hideAllSections() {
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// Navigate to history
function viewHistory() {
    window.location.href = '/history';
}

// Utility: Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Service worker registration (for offline support)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('Service Worker registered:', registration);
            })
            .catch(error => {
                console.log('Service Worker registration failed:', error);
            });
    });
}
