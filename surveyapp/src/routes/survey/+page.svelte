<script lang="ts">
    let data = [
        "4980-Balcony.jpg",
        "4980-City.jpg",
        "4980-City5.jpg",
        "4980-Football.jpg",
        "4980-IATL.jpg",
        "4980-Linquistic.jpg",
        "4980-Park.jpg",
        "4980-Seamans2.jpg",
        "4980-eng.jpg",
        "4980-law.jpg",
        "4980-Biz.jpg",
        "4980-City2.jpg",
        "4980-City6.jpg",
        "4980-Forest.jpg",
        "4980-IMU.jpg",
        "4980-MF-Outer.jpg",
        "4980-River.jpg",
        "4980-Staples.jpg",
        "4980-epp.jpg",
        "4980-library.jpg",
        "4980-Bridge.jpg",
        "4980-City4.jpg",
        "4980-Downtown3.jpg",
        "4980-GradHotel.jpg",
        "4980-IMU2.jpg",
        "4980-Mall.jpg",
        "4980-Seamans.jpg",
        "4980-Westside.jpg",
        "4980-iowabook.jpg",
        "4980-med.jpg"
    ];

    type Relationship = {
        img1: string;
        img2: string;
        related: boolean;
        reason: string;
    };

    // Create anonymous mapping for images
    let imageMapping: Map<string, string> = new Map();
    
    function initializeImageMapping() {
        imageMapping = new Map();
        data.forEach((img, index) => {
            imageMapping.set(img, `Image_${(index + 1).toString().padStart(2, '0')}`);
        });
    }

    function getAnonymousName(img: string): string {
        return imageMapping.get(img) || img;
    }

    // State variables
    let surveyStarted = false;
    let surveyComplete = false;
    let currentImageIndex = 0;
    let selectedImages: Set<string> = new Set();
    let mode: 'selection' | 'reasons' = 'selection'; // selection or reasons entry
    let reasonInputs: Map<string, string> = new Map();
    let allRelationships: Relationship[] = [];
    let copySuccess = false;

    // Computed values
    $: currentImage = data[currentImageIndex];
    $: progress = data.length > 0 ? (((currentImageIndex + 1) / data.length) * 100).toFixed(1) : 0;
    
    // Get images that should be shown in the selection grid
    $: availableImages = data.filter(img => {
        // Don't show the current image
        if (img === currentImage) return false;
        
        // Don't show images that already have a relationship with current image
        const hasRelationship = allRelationships.some(rel => 
            (rel.img1 === currentImage && rel.img2 === img) ||
            (rel.img1 === img && rel.img2 === currentImage)
        );
        
        return !hasRelationship;
    });

    function startSurvey() {
        initializeImageMapping();
        surveyStarted = true;
        currentImageIndex = 0;
        selectedImages = new Set();
        mode = 'selection';
        reasonInputs = new Map();
        allRelationships = [];
        surveyComplete = false;
        copySuccess = false;
    }

    function toggleImageSelection(image: string) {
        if (selectedImages.has(image)) {
            selectedImages.delete(image);
            selectedImages = selectedImages; // Trigger reactivity
        } else {
            selectedImages.add(image);
            selectedImages = selectedImages; // Trigger reactivity
        }
    }

    function proceedToReasons() {
        if (selectedImages.size === 0) {
            // No images selected, move to next image
            moveToNextImage();
        } else {
            // Initialize reason inputs for selected images
            reasonInputs = new Map();
            selectedImages.forEach(img => {
                reasonInputs.set(img, "");
            });
            mode = 'reasons';
        }
    }

    function updateReason(image: string, reason: string) {
        reasonInputs.set(image, reason);
        reasonInputs = reasonInputs; // Trigger reactivity
    }

    function submitReasons() {
        let hasEmptyReason = false;
        
        // Check if all reasons are filled
        for (const [img, reason] of reasonInputs) {
            if (reason.trim() === "") {
                hasEmptyReason = true;
                break;
            }
        }

        if (hasEmptyReason) {
            alert("Please provide a reason for all selected relationships.");
            return;
        }

        // Save all relationships
        selectedImages.forEach(img => {
            allRelationships.push({
                img1: currentImage,
                img2: img,
                related: true,
                reason: reasonInputs.get(img) || ""
            });
        });

        // Move to next image
        moveToNextImage();
    }

    function skipImage() {
        // User chose not to select any images
        moveToNextImage();
    }

    function moveToNextImage() {
        currentImageIndex++;
        selectedImages = new Set();
        mode = 'selection';
        reasonInputs = new Map();

        if (currentImageIndex >= data.length) {
            surveyComplete = true;
        }
    }

    function copyResultsToClipboard() {
        // Create CSV format
        let csvContent = "Image 1,Image 2,Related,Reason\n";
        allRelationships.forEach(rel => {
            const img1 = getAnonymousName(rel.img1);
            const img2 = getAnonymousName(rel.img2);
            const reason = rel.reason.replace(/"/g, '""'); // Escape quotes
            csvContent += `"${img1}","${img2}","Yes","${reason}"\n`;
        });

        // Copy to clipboard
        navigator.clipboard.writeText(csvContent).then(() => {
            copySuccess = true;
            setTimeout(() => {
                copySuccess = false;
            }, 3000);
        }).catch(err => {
            alert('Failed to copy to clipboard: ' + err);
        });
    }
</script>

<div class="container">
    <h1>Welcome to the ESN Survey!</h1>
    <p>Today, you will be categorizing image relationships.</p>

    {#if !surveyStarted}
        <div class="intro">
            <p>This survey will show you one image at a time. For each image, you'll:</p>
            <ol>
                <li><strong>Select</strong> all images from the collection that are semantically related to it</li>
                <li><strong>Provide a reason</strong> for each selected relationship</li>
                <li><strong>Continue</strong> to the next image</li>
            </ol>
            <p>Total images to evaluate: <strong>{data.length}</strong></p>
            <button class="start-button" on:click={startSurvey}>Start Survey</button>
        </div>
    {:else if surveyStarted && !surveyComplete}
        <div class="survey-progress">
            <p>Image: {currentImageIndex + 1} / {data.length} ({progress}%)</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress}%"></div>
            </div>
        </div>

        {#if mode === 'selection'}
            <div class="selection-mode">
                <h2>Select images related to:</h2>
                
                <div class="main-layout">
                <div class="left-panel">
                    <div class="current-image-container">
                        <img src="/workdata/{currentImage}" alt={currentImage} class="current-image" />
                        <p class="image-label"><strong>{getAnonymousName(currentImage)}</strong></p>
                    </div>
                </div>                    <div class="right-panel">
                        <h3>Select related images ({selectedImages.size} selected):</h3>
                        {#if availableImages.length === 0}
                            <p class="no-images">All remaining images have already been evaluated with this image.</p>
                        {:else}
                            <div class="image-grid">
                                {#each availableImages as image}
                                    <div 
                                        class="grid-image-container {selectedImages.has(image) ? 'selected' : ''}"
                                        on:click={() => toggleImageSelection(image)}
                                        role="button"
                                        tabindex="0"
                                        on:keypress={(e) => e.key === 'Enter' && toggleImageSelection(image)}
                                    >
                                        <img src="/workdata/{image}" alt={image} class="grid-image" />
                                        <p class="grid-image-label">{getAnonymousName(image)}</p>
                                        {#if selectedImages.has(image)}
                                            <div class="selected-badge">✓</div>
                                        {/if}
                                    </div>
                                {/each}
                            </div>
                        {/if}
                    </div>
                </div>

                <div class="button-section">
                    <button class="skip-button" on:click={skipImage}>
                        Skip (No Related Images)
                    </button>
                    <button 
                        class="continue-button" 
                        on:click={proceedToReasons}
                        disabled={selectedImages.size === 0}
                    >
                        Continue to Reasons ({selectedImages.size})
                    </button>
                </div>
            </div>
        {:else if mode === 'reasons'}
            <div class="reasons-mode">
                <h2>Provide reasons for relationships</h2>
                <p class="base-image-label">Base image: <strong>{getAnonymousName(currentImage)}</strong></p>

                <div class="reasons-list">
                    {#each Array.from(selectedImages) as image}
                        <div class="reason-item">
                            <div class="reason-images">
                                <img src="/workdata/{currentImage}" alt={currentImage} class="reason-thumb" />
                                <span class="arrow">→</span>
                                <img src="/workdata/{image}" alt={image} class="reason-thumb" />
                            </div>
                            <div class="reason-input-group">
                                <label for="reason-{image}">{getAnonymousName(image)}</label>
                                <textarea
                                    id="reason-{image}"
                                    value={reasonInputs.get(image) || ""}
                                    on:input={(e) => updateReason(image, e.currentTarget.value)}
                                    placeholder="Why are these images related?"
                                    rows="3"
                                ></textarea>
                            </div>
                        </div>
                    {/each}
                </div>

                <div class="button-section">
                    <button class="back-button" on:click={() => mode = 'selection'}>
                        ← Back to Selection
                    </button>
                    <button class="submit-button" on:click={submitReasons}>
                        Submit & Continue
                    </button>
                </div>
            </div>
        {/if}
    {:else if surveyComplete}
        <div class="results">
            <h2>Survey Complete!</h2>
            <p>Thank you for completing the survey. Here are all the relationships you identified:</p>
            
            <div class="copy-button-container">
                <button class="copy-button" on:click={copyResultsToClipboard}>
                    {copySuccess ? '✓ Copied!' : '📋 Copy Results to Clipboard'}
                </button>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Image 1</th>
                            <th>Image 2</th>
                            <th>Related</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each allRelationships as rel}
                            <tr>
                                <td>{getAnonymousName(rel.img1)}</td>
                                <td>{getAnonymousName(rel.img2)}</td>
                                <td class="related-yes">Yes</td>
                                <td>{rel.reason}</td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>

            <div class="summary">
                <p><strong>Total Relationships:</strong> {allRelationships.length}</p>
                <p><strong>Images Evaluated:</strong> {data.length}</p>
                <p><strong>Average Relationships per Image:</strong> {(allRelationships.length / data.length).toFixed(2)}</p>
            </div>
        </div>
    {/if}
</div>

<style>
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    h1 {
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }

    h2 {
        color: #495057;
        text-align: center;
        margin: 1.5rem 0;
    }

    h3 {
        color: #495057;
        margin-bottom: 1rem;
    }

    .intro {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        margin: 2rem 0;
        text-align: center;
    }

    .intro ol {
        text-align: left;
        max-width: 600px;
        margin: 1rem auto;
    }

    .start-button {
        background: #007bff;
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.3s;
        margin-top: 1rem;
    }

    .start-button:hover {
        background: #0056b3;
    }

    .survey-progress {
        margin: 2rem 0;
        text-align: center;
    }

    .progress-bar {
        width: 100%;
        height: 30px;
        background: #e9ecef;
        border-radius: 15px;
        overflow: hidden;
        margin-top: 1rem;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #007bff, #0056b3);
        transition: width 0.3s ease;
    }

    /* Selection Mode Styles */
    .selection-mode {
        margin: 2rem 0;
    }

    .main-layout {
        display: flex;
        gap: 2rem;
        margin: 2rem 0;
        align-items: flex-start;
    }

    .left-panel {
        flex: 0 0 400px;
        position: sticky;
        top: 2rem;
    }

    .current-image-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .current-image {
        max-width: 100%;
        max-height: 400px;
        border: 3px solid #007bff;
        border-radius: 8px;
    }

    .image-label {
        margin-top: 1rem;
        font-size: 1rem;
        color: #495057;
    }

    .right-panel {
        flex: 1;
    }

    .no-images {
        text-align: center;
        color: #6c757d;
        font-style: italic;
        padding: 2rem;
    }

    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }

    .grid-image-container {
        position: relative;
        background: #fff;
        border: 3px solid #dee2e6;
        border-radius: 8px;
        padding: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
        text-align: center;
    }

    .grid-image-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    .grid-image-container.selected {
        border-color: #28a745;
        background: #d4edda;
    }

    .grid-image {
        width: 100%;
        height: 220px;
        object-fit: cover;
        border-radius: 4px;
    }

    .grid-image-label {
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #6c757d;
        word-break: break-all;
    }

    .selected-badge {
        position: absolute;
        top: 0.75rem;
        right: 0.75rem;
        background: #28a745;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.5rem;
    }

    /* Reasons Mode Styles */
    .reasons-mode {
        margin: 2rem 0;
    }

    .base-image-label {
        text-align: center;
        font-size: 1.1rem;
        color: #495057;
        margin: 1rem 0;
    }

    .reasons-list {
        max-width: 900px;
        margin: 2rem auto;
    }

    .reason-item {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .reason-images {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .reason-thumb {
        width: 100px;
        height: 100px;
        object-fit: cover;
        border: 2px solid #dee2e6;
        border-radius: 4px;
    }

    .arrow {
        font-size: 2rem;
        color: #007bff;
        font-weight: bold;
    }

    .reason-input-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .reason-input-group label {
        font-weight: 600;
        color: #495057;
    }

    .reason-input-group textarea {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        border: 2px solid #ced4da;
        border-radius: 4px;
        font-family: inherit;
        resize: vertical;
        transition: border-color 0.3s;
    }

    .reason-input-group textarea:focus {
        outline: none;
        border-color: #28a745;
    }

    /* Button Styles */
    .button-section {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }

    .skip-button,
    .continue-button,
    .back-button,
    .submit-button {
        padding: 1rem 2rem;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s;
        min-width: 180px;
    }

    .skip-button {
        background: #6c757d;
        color: white;
    }

    .skip-button:hover {
        background: #5a6268;
    }

    .continue-button {
        background: #007bff;
        color: white;
    }

    .continue-button:hover:not(:disabled) {
        background: #0056b3;
    }

    .continue-button:disabled {
        background: #ccc;
        cursor: not-allowed;
        opacity: 0.6;
    }

    .back-button {
        background: #6c757d;
        color: white;
    }

    .back-button:hover {
        background: #5a6268;
    }

    .submit-button {
        background: #28a745;
        color: white;
    }

    .submit-button:hover {
        background: #218838;
    }

    /* Results Styles */
    .results {
        margin: 2rem 0;
    }

    .results h2 {
        color: #28a745;
        text-align: center;
    }

    .table-container {
        overflow-x: auto;
        margin: 2rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    table {
        width: 100%;
        border-collapse: collapse;
        background: white;
    }

    thead {
        background: #343a40;
        color: white;
    }

    th, td {
        padding: 1rem;
        text-align: left;
        border-bottom: 1px solid #dee2e6;
    }

    th {
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
    }

    tbody tr:hover {
        background: #f8f9fa;
    }

    .related-yes {
        color: #28a745;
        font-weight: bold;
    }

    .summary {
        background: #e9ecef;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 2rem;
        text-align: center;
    }

    .summary p {
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }

    .copy-button-container {
        text-align: center;
        margin: 2rem 0;
    }

    .copy-button {
        background: #007bff;
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .copy-button:hover {
        background: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    @media (max-width: 1024px) {
        .main-layout {
            flex-direction: column;
        }

        .left-panel {
            flex: 1;
            position: static;
            width: 100%;
        }

        .image-grid {
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        }
    }

    @media (max-width: 768px) {
        .container {
            padding: 1rem;
        }

        .image-grid {
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 0.5rem;
        }

        .button-section {
            flex-direction: column;
        }

        .skip-button,
        .continue-button,
        .back-button,
        .submit-button {
            width: 100%;
        }
    }
</style>

