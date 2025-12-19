let mediaRecorder;
let audioChunks = [];
let recordingStartTime;
let timerInterval;
let currentTimestamp;
let currentDetections = []; // Store current YOLO detections
let isMarkingMode = false;
let markedObjects = [];
let tempMarkerPosition = null;
let currentOcrData = null;

// DOM Elements
const recordButton = document.getElementById('recordButton');
const markObjectsButton = document.getElementById('markObjectsButton');
const recordingTimer = document.getElementById('recordingTimer');
const transcriptionDiv = document.getElementById('transcription');
const refinedTranscription = document.getElementById('refinedTranscription');
const saveRefinedButton = document.getElementById('saveRefined');
const currentImage = document.getElementById('currentImage');
const ocrButton = document.getElementById('ocrButton');
const ocrResults = document.getElementById('ocrResults');

// Add D3.js library loading
const script = document.createElement('script');
script.src = 'https://d3js.org/d3.v7.min.js';
script.onload = () => {
    console.log('D3.js loaded successfully');
    init();
};
document.head.appendChild(script);
const objectDialog = document.getElementById('objectDialog');
const objectName = document.getElementById('objectName');
const saveMarkObject = document.getElementById('saveMarkObject');
const cancelMarkObject = document.getElementById('cancelMarkObject');
const objectList = document.getElementById('objectList');
const markersContainer = document.getElementById('markersContainer');

// Initialize
async function init() {
    try {
        console.log('Starting initialization...');
        
        // Create loading indicator first
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.textContent = 'Loading image...';
        loadingDiv.style.position = 'absolute';
        loadingDiv.style.top = '50%';
        loadingDiv.style.left = '50%';
        loadingDiv.style.transform = 'translate(-50%, -50%)';
        loadingDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
        loadingDiv.style.padding = '10px';
        loadingDiv.style.borderRadius = '5px';
        document.querySelector('.image-container').appendChild(loadingDiv);
        console.log('Loading indicator created');

        // Load initial image
        console.log('Fetching image from server...');
        const response = await fetch('/api/get-image');
        const data = await response.json();
        console.log('Received image data:', data);
        
        // Set up image load handler before setting src
        currentImage.onload = async () => {
            console.log('Image loaded successfully');
            // Change loading text to indicate processing
            loadingDiv.textContent = 'Processing image...';
            
            try {
                console.log('Starting background processing...');
                // Run object detection first
                await runObjectDetection();
                console.log('Object detection completed');
                
                // Then run pre-segmentation
                // await runPreSegmentation();
                console.log('Pre-segmentation completed');
            } catch (error) {
                console.error('Error processing image:', error);
            } finally {
                // Remove loading indicator
                loadingDiv.remove();
                console.log('Loading indicator removed');
            }
        };

        // Set image source after setting up onload handler
        console.log('Setting image source:', `/static/images/${data.image}`);
        const imageFilename = data.image;
    currentImage.src = `/static/images/${imageFilename}`;
    
    // Load existing marked objects
    const objectsResponse = await fetch(`/api/get-objects/${imageFilename}`);
    const objectsData = await objectsResponse.json();
    markedObjects = objectsData.objects || [];
    
    // Display existing markers
    markedObjects.forEach(object => {
        addMarkerToImage(object);
    });
    updateObjectList();
        
        // Set up click handlers for image segmentation and object marking
        currentImage.addEventListener('click', handleImageClick);
        console.log('Initialization completed');
    } catch (error) {
        console.error('Error initializing application:', error);
    }
}

// Run object detection on current image
async function runObjectDetection() {
    try {
        // Get image as base64
        const imageBlob = await fetch(currentImage.src).then(res => res.blob());
        const reader = new FileReader();
        
        const base64data = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(imageBlob);
        });
        
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64data
            })
        });
        
        if (!response.ok) {
            throw new Error('Object detection request failed');
        }
        
        const data = await response.json();
        if (!data.detections) {
            throw new Error('No detections in response');
        }
        
        currentDetections = data.detections;
        console.log('Object detections:', currentDetections);
    } catch (err) {
        console.error('Error running object detection:', err);
        currentDetections = []; // Reset detections on error
    }
}

// Check if a point is inside a bounding box
function isPointInBox(point, box) {
    const [x1, y1, x2, y2] = box;
    const [x, y] = point;
    const isInside = x >= x1 && x <= x2 && y >= y1 && y <= y2;
    console.log(`Checking point (${x}, ${y}) against box [${x1}, ${y1}, ${x2}, ${y2}]: ${isInside}`);
    return isInside;
}

// Get object name at click point
function getObjectAtPoint(point) {
    console.log('Checking detections at point:', point);
    console.log('Current detections:', currentDetections);
    
    for (const detection of currentDetections) {
        if (isPointInBox(point, detection.box)) {
            console.log('Found object:', detection);
            return {
                name: detection.class,
                confidence: detection.confidence
            };
        }
    }
    console.log('No object found at point');
    return null;
}

// Recording functions
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioForTranscription(audioBlob);
        };
        
        audioChunks = [];
        mediaRecorder.start();
        recordingStartTime = Date.now();
        updateRecordingTimer();
        
        recordButton.textContent = 'Stop Recording';
        recordButton.classList.add('recording');
        recordingTimer.classList.remove('hidden');
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Error accessing microphone. Please ensure you have granted microphone permissions.');
    }
}

function stopRecording() {
    mediaRecorder.stop();
    clearInterval(timerInterval);
    
    recordButton.textContent = 'Start Recording';
    recordButton.classList.remove('recording');
    recordingTimer.classList.add('hidden');
}

function updateRecordingTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        recordingTimer.textContent = `${minutes}:${seconds}`;
        
        // Stop recording after 90 seconds
        if (elapsed >= 90) {
            stopRecording();
        }
    }, 1000);
}

// Transcription functions
async function sendAudioForTranscription(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('image_filename', currentImage.src.split('/').pop());
    
    // Add OCR text if available
    if (currentOcrData) {
        const ocrText = currentOcrData.map(item => item.text).join('\n');
        formData.append('ocr_text', ocrText);
    }
    
    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        currentTimestamp = data.timestamp;
        
        // Display original transcriptions
        if (data.segmented_transcriptions && data.segmented_transcriptions.length > 0) {
            const fullTranscription = data.segmented_transcriptions
                .map(segment => segment.transcription)
                .join('\n');
            
            transcriptionDiv.textContent = fullTranscription;
        } else {
            transcriptionDiv.textContent = "No transcription available.";
        }

        // Display refined transcription if available
        if (data.refined_transcription) {
            refinedTranscription.value = data.refined_transcription;
        } else {
            refinedTranscription.value = "GPT refinement not available. Please try again.";
        }
    } catch (err) {
        console.error('Error sending audio for transcription:', err);
        alert('Error processing audio. Please try again.');
    }
}

// Object Marking Functions
function toggleMarkingMode() {
    isMarkingMode = !isMarkingMode;
    markObjectsButton.classList.toggle('active');
    markObjectsButton.textContent = isMarkingMode ? 'Stop Marking' : 'Mark Objects';
    
    // Toggle cursor style on the image container
    const imageContainer = currentImage.parentElement;
    imageContainer.style.cursor = isMarkingMode ? 'crosshair' : 'default';

    // If stopping marking mode, export the data
    if (!isMarkingMode) {
        exportMarkedObjects();
    }
}

async function handleImageClick(event) {
    const rect = currentImage.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert to image coordinates
    const clickX = (x / rect.width) * currentImage.naturalWidth;
    const clickY = (y / rect.height) * currentImage.naturalHeight;
    
    if (isMarkingMode) {
        showObjectDialog(clickX, clickY);
        return;
    }
    
    // Check for object detection first
    const object = getObjectAtPoint([clickX, clickY]);
    const tooltip = d3.select(".segment-tooltip");
    
    if (object) {
        // Display object detection result with confidence score
        tooltip.html(`${object.name} (${(object.confidence * 100).toFixed(1)}%)`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px")
            .style("display", "block");
    } else {
        // Display "No object detected" with 0% confidence
        tooltip.html(`No object detected (0.0%)`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px")
            .style("display", "block");
    }
    
    // Hide tooltip after 2 seconds
    setTimeout(() => {
        tooltip.style("display", "none");
    }, 2000);
    
    // Only proceed with segmentation if recording
    if (!mediaRecorder || mediaRecorder.state !== 'recording') {
        console.log('Not recording, segmentation click ignored.');
        return;
    }

    // Original segmentation code...
    const imageX = (x / rect.width) * currentImage.naturalWidth;
    const imageY = (y / rect.height) * currentImage.naturalHeight;
    
    // Get image as base64
    const imageBlob = await fetch(currentImage.src).then(res => res.blob());
    const reader = new FileReader();
    reader.readAsDataURL(imageBlob);
    
    reader.onloadend = async () => {
        const base64data = reader.result;
        
        try {
            const response = await fetch('/api/segment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64data,
                    x: imageX,
                    y: imageY,
                    regions: window.currentRegions || []  // Pass current regions for background check
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                console.log('Segment request failed:', error.error);
                return;
            }
            
            const data = await response.json();
            
            // Add the new segment to the visualization
            const overlay = document.getElementById('segmentationOverlay');
            if (overlay) {
                const svg = d3.select(overlay).select('svg');
                
                // If this is a background segment, remove previous background segments
                if (data.class === 'background') {
                    // Remove previous background segments from SVG
                    svg.selectAll(".segment")
                        .filter(function() {
                            return d3.select(this).attr("data-class") === "background";
                        })
                        .remove();
                    
                    // Remove previous background segments from currentRegions
                    if (window.currentRegions) {
                        window.currentRegions = window.currentRegions.filter(region => region.class !== 'background');
                    }
                }
                
                // Create a new group for the segment
                const group = svg.append("g")
                    .attr("class", "segment")
                    .attr("data-id", data.id)
                    .attr("data-class", data.class)
                    .attr("data-score", data.score)
                    .style("opacity", 0.7)
                    .style("pointer-events", "auto")
                    .style("cursor", "pointer");
                
                // Draw polygons for the segment
                data.polygons.forEach(polygon => {
                    // Ensure polygon is closed
                    if (polygon.length > 0 && polygon[0] !== polygon[polygon.length - 1]) {
                        polygon.push(polygon[0]);
                    }
                    
                    // Create polygon with explicit styling
                    group.append("polygon")
                        .attr("points", polygon.map(p => `${p[0]},${p[1]}`).join(" "))
                        .style("fill", `rgba(${data.color.join(",")}, 0.5)`)
                        .style("stroke", `rgb(${data.color.join(",")})`)
                        .style("stroke-width", "2px")
                        .style("vector-effect", "non-scaling-stroke")
                        .style("pointer-events", "auto");
                });
                
                // Add hover effects
                group.on("mouseover", function(event) {
                    // Highlight the hovered segment by increasing opacity of its pixels
                    d3.select(this).selectAll("rect")
                        .style("opacity", 0.9); // Adjust opacity for hover
                    
                    svg.selectAll(".segment")
                        .filter(other => other.id !== data.id)
                        .style("opacity", 0.4); // Dim other segments
                    
                    let tooltipText = `${data.class}`;
                    if (data.score) {
                        tooltipText += ` (${(data.score * 100).toFixed(1)}%)`;
                    }
                    
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.html(tooltipText)
                })
                .on("mousemove", function(event) {
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mouseout", function() {
                    svg.selectAll(".segment")
                        .style("opacity", 0.7); // Restore opacity of all segments
                    
                    // Restore opacity of pixels in the hovered segment
                    d3.select(this).selectAll("rect")
                        .style("opacity", 0.7); // Adjust opacity back to normal
                    
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.style("display", "none");
                });
                
                // Store the new region
                if (!window.currentRegions) {
                    window.currentRegions = [];
                }
                window.currentRegions.push(data);
            }
        } catch (err) {
            console.error('Error processing image segmentation:', err);
        }
    };
}

function showObjectDialog(x, y) {
    objectDialog.classList.remove('hidden');
    tempMarkerPosition = { x, y };
    objectName.value = '';
    objectName.focus();
}

function saveObject() {
    const name = objectName.value.trim().toLowerCase(); // Convert to lowercase
    if (name && tempMarkerPosition) {
        // Check if object with this name already exists
        const existingObjects = markedObjects.filter(obj => obj.name === name);
        const count = existingObjects.length + 1;
        
        const object = {
            id: Date.now(),
            name: name,
            x: tempMarkerPosition.x,
            y: tempMarkerPosition.y,
            count: count
        };
        
        markedObjects.push(object);
        addMarkerToImage(object);
        updateObjectList();
        saveMarkedObjects();
        
        hideObjectDialog();
    }
}

function addMarkerToImage(object) {
    const marker = document.createElement('div');
    marker.className = 'marker';
    
    // Calculate position relative to the image
    const xPercent = (object.x / currentImage.naturalWidth) * 100;
    const yPercent = (object.y / currentImage.naturalHeight) * 100;
    
    marker.style.left = `${xPercent}%`;
    marker.style.top = `${yPercent}%`;
    marker.setAttribute('data-object-id', object.id);
    
    // Add label with just the name
    const label = document.createElement('div');
    label.className = 'marker-label';
    label.textContent = object.name;
    marker.appendChild(label);
    
    markersContainer.appendChild(marker);
}

function updateObjectList() {
    // Group objects by name to show counts
    const objectCounts = {};
    markedObjects.forEach(obj => {
        if (!objectCounts[obj.name]) {
            objectCounts[obj.name] = 1;
        } else {
            objectCounts[obj.name]++;
        }
    });
    
    // Create HTML for object list
    const objectListHTML = Object.entries(objectCounts).map(([name, count]) => `
        <div class="flex justify-between items-center py-1">
            <span>${name} (${count})</span>
            <div class="flex space-x-2">
                <button onclick="removeObjectsByName('${name}')" class="text-red-500 hover:text-red-700">
                    ×
                </button>
            </div>
        </div>
    `).join('');
    
    objectList.innerHTML = objectListHTML;
}

function removeObjectsByName(name) {
    // Remove all markers with this name
    markedObjects.filter(obj => obj.name === name).forEach(obj => {
        const marker = markersContainer.querySelector(`[data-object-id="${obj.id}"]`);
        if (marker) {
            marker.remove();
        }
    });
    
    // Remove objects from array
    markedObjects = markedObjects.filter(obj => obj.name !== name);
    
    // Update UI and save
    updateObjectList();
    saveMarkedObjects();
}

async function saveMarkedObjects() {
    try {
        const response = await fetch('/api/save-objects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_filename: currentImage.src.split('/').pop(),
                objects: markedObjects
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to save marked objects');
        }
    } catch (err) {
        console.error('Error saving marked objects:', err);
    }
}

function hideObjectDialog() {
    objectDialog.classList.add('hidden');
    tempMarkerPosition = null;
}

function cancelObjectMarking() {
    hideObjectDialog();
}

// Add event listeners
markObjectsButton.addEventListener('click', toggleMarkingMode);
saveMarkObject.addEventListener('click', saveObject);
cancelMarkObject.addEventListener('click', cancelObjectMarking);
objectName.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        saveObject();
    }
});

// Add pre-segmentation function
async function runPreSegmentation() {
    try {
        console.log('Starting pre-segmentation...');
        // Get image as base64
        const imageBlob = await fetch(currentImage.src).then(res => res.blob());
        const reader = new FileReader();
        
        // Convert blob to base64
        const base64data = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(imageBlob);
        });
        
        console.log('Sending pre-segmentation request...');
        const response = await fetch('/api/pre-segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64data
            })
        });
        
        const data = await response.json();
        console.log('Received segmentation data:', data.regions ? data.regions.length : 0, 'regions');
        
        if (data.regions) {
            // Wait for next frame to ensure DOM is ready
            await new Promise(resolve => requestAnimationFrame(resolve));
            
            // Create or update segmentation overlay
            let overlay = document.getElementById('segmentationOverlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'segmentationOverlay';
                overlay.className = 'segmentation-overlay';
                overlay.style.position = 'absolute';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.pointerEvents = 'none';
                overlay.style.zIndex = '1000';
                document.querySelector('.image-container').appendChild(overlay);
            }
            
            // Clear previous visualization
            overlay.innerHTML = '';
            
            // Get image dimensions and position
            const rect = currentImage.getBoundingClientRect();
            const imgWidth = currentImage.naturalWidth;
            const imgHeight = currentImage.naturalHeight;
            console.log('Image dimensions:', imgWidth, 'x', imgHeight);
            console.log('Image position:', rect);
            
            // Create SVG with explicit dimensions and position
            const svg = d3.select(overlay)
                .append("svg")
                .attr("width", rect.width)
                .attr("height", rect.height)
                .attr("viewBox", `0 0 ${imgWidth} ${imgHeight}`)
                .style("position", "absolute")
                .style("top", "0")
                .style("left", "0")
                .style("width", "100%")
                .style("height", "100%")
                .style("pointer-events", "none");
            
            // Create tooltip div
            const tooltip = d3.select("body")
                .append("div")
                .attr("class", "segment-tooltip")
                .style("position", "absolute")
                .style("background-color", "rgba(0, 0, 0, 0.8)")
                .style("color", "white")
                .style("padding", "8px")
                .style("border-radius", "4px")
                .style("font-size", "14px")
                .style("pointer-events", "none")
                .style("z-index", "1000")
                .style("display", "none");
            
            console.log('Drawing segments...');
            // Create a group for each region
            const regionGroups = svg.selectAll("g")
                .data(data.regions)
                .enter()
                .append("g")
                .attr("class", "segment")
                .attr("data-id", d => d.id)
                .attr("data-class", d => d.class)
                .attr("data-score", d => d.score)
                .style("opacity", 0.7)  // Make segments more visible
                .style("pointer-events", "auto")
                .style("cursor", "pointer");
            
            // Draw polygons for each region
            regionGroups.each(function(region) {
                console.log('Drawing region:', region);
                const group = d3.select(this);
                // Draw mask pixels as rectangles
                const mask = region.mask;
                const color = region.color;

                for (let y = 0; y < mask.length; y++) {
                    for (let x = 0; x < mask[0].length; x++) {
                        if (mask[y][x] === 1) {
                            group.append("rect")
                                .attr("x", x)
                                .attr("y", y)
                                .attr("width", 1)
                                .attr("height", 1)
                                .style("fill", `rgb(${color.join(",")})`)
                                .style("opacity", 0.7) // Adjust opacity as needed
                                .style("pointer-events", "auto");
                        }
                    }
                }
            });
            
            // Add hover effects
            regionGroups
                .on("mouseover", function(event, d) {
                    // Highlight the hovered segment using a filter
                    d3.select(this)
                        .style("filter", "brightness(1.2)"); // Apply a brightness filter on hover
                    
                    svg.selectAll(".segment")
                        .filter(other => other.id !== d.id)
                        .style("opacity", 0.4); // Dim other segments
                    
                    // Find the center of the mask (optional - could use bounding box or just the event point)
                    const mask = d.mask;
                    let sumX = 0;
                    let sumY = 0;
                    let count = 0;
                    for (let y = 0; y < mask.length; y++) {
                        for (let x = 0; x < mask[0].length; x++) {
                            if (mask[y][x] === 1) {
                                sumX += x;
                                sumY += y;
                                count++;
                            }
                        }
                    }
                    
                    let bestMatch = null;
                    if (count > 0) {
                        const maskCenterX = sumX / count;
                        const maskCenterY = sumY / count;
                        
                        // Find overlapping detection with highest confidence
                        for (const detection of currentDetections) {
                            // Check if mask center is within detection bounding box
                            if (isPointInBox([maskCenterX, maskCenterY], detection.box)) {
                                if (!bestMatch || detection.confidence > bestMatch.confidence) {
                                    bestMatch = detection;
                                }
                            }
                        }
                    }
                    
                    let tooltipText;
                    if (bestMatch) {
                        // If object detection found, show its class and confidence
                        tooltipText = `${bestMatch.class} (${(bestMatch.confidence * 100).toFixed(1)}%)`;
                    } else {
                        // If no object detection, show segmentation class without score
                         tooltipText = `${d.class}`;
                    }

                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.html(tooltipText)
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mousemove", function(event) {
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mouseout", function() {
                    svg.selectAll(".segment")
                        .style("opacity", 0.7); // Restore opacity of all segments
                    
                    // Remove the filter from the hovered segment
                    d3.select(this)
                        .style("filter", null); // Remove the filter on mouseout
                    
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.style("display", "none");
                });
            
            console.log('Segmentation visualization complete');
        }
    } catch (err) {
        console.error('Error running pre-segmentation:', err);
    }
}

async function exportMarkedObjects() {
    try {
        // Group objects by name with their coordinates and count
        const objectSummary = {};
        let currentId = 1;  // Start IDs from 1
        
        markedObjects.forEach(obj => {
            if (!objectSummary[obj.name]) {
                objectSummary[obj.name] = {
                    id: currentId++,
                    name: obj.name,
                    coordinates: [{x: obj.x, y: obj.y}],
                    count: 1
                };
            } else {
                objectSummary[obj.name].coordinates.push({x: obj.x, y: obj.y});
                objectSummary[obj.name].count++;
            }
        });
        
        // Convert to array format
        const exportObjects = Object.values(objectSummary);
        
        const response = await fetch('/api/export-objects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_filename: currentImage.src.split('/').pop(),
                objects: exportObjects,
                timestamp: new Date().toISOString()
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to export marked objects');
        }
        
        const data = await response.json();
        if (data.success) {
            console.log('Objects exported successfully to:', data.filename);
        }
    } catch (err) {
        console.error('Error exporting marked objects:', err);
    }
}

// Add OCR-related functions
async function performOCR() {
    try {
        // Show loading state
        ocrButton.disabled = true;
        ocrButton.textContent = 'Processing OCR...';
        
        // Get current image as base64
        const imageBlob = await fetch(currentImage.src).then(res => res.blob());
        const reader = new FileReader();
        
        const base64data = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(imageBlob);
        });
        
        // Send to backend
        const response = await fetch('/api/ocr', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64data
            })
        });
        
        if (!response.ok) {
            throw new Error('OCR request failed');
        }
        
        const data = await response.json();
        currentOcrData = data.ocr_data;
        
        // Display OCR results
        displayOCRResults(currentOcrData);
        
        // Highlight text regions on image
        highlightOCRRegions(currentOcrData);
        
    } catch (err) {
        console.error('Error performing OCR:', err);
        alert('Error performing OCR. Please try again.');
    } finally {
        // Reset button state
        ocrButton.disabled = false;
        ocrButton.textContent = 'Perform OCR';
    }
}

function displayOCRResults(ocrData) {
    if (!ocrData || ocrData.length === 0) {
        ocrResults.innerHTML = '<p class="text-gray-500">No text detected in the image.</p>';
        return;
    }
    
    const resultsHTML = ocrData.map(item => `
        <div class="ocr-result p-2 border-b border-gray-200">
            <p class="font-medium">${item.text}</p>
            <p class="text-sm text-gray-500">Confidence: ${(item.confidence * 100).toFixed(1)}%</p>
        </div>
    `).join('');
    
    ocrResults.innerHTML = `
        <div class="bg-white rounded-lg shadow p-4">
            <h3 class="text-lg font-semibold mb-4">Detected Text</h3>
            ${resultsHTML}
        </div>
    `;
}

function highlightOCRRegions(ocrData) {
    // Remove existing highlights
    d3.select('#ocrOverlay').remove();
    
    if (!ocrData || ocrData.length === 0) return;
    
    const imageContainer = document.querySelector('.image-container');
    const overlay = d3.select(imageContainer)
        .append('div')
        .attr('id', 'ocrOverlay')
        .style('position', 'absolute')
        .style('top', '0')
        .style('left', '0')
        .style('width', '100%')
        .style('height', '100%')
        .style('pointer-events', 'none');
    
    const svg = overlay.append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .style('position', 'absolute');
    
    ocrData.forEach(item => {
        const points = item.bbox.map(point => 
            `${point[0] * 100}% ${point[1] * 100}%`
        ).join(',');
        
        svg.append('polygon')
            .attr('points', points)
            .style('fill', 'rgba(255, 255, 0, 0.2)')
            .style('stroke', 'rgba(255, 200, 0, 0.8)')
            .style('stroke-width', '2px');
    });
}

// Add event listener for OCR button
ocrButton.addEventListener('click', performOCR);