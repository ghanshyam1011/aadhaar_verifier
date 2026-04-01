// =============================================
// AadhaarCheck Frontend Controller
// =============================================

// Backend API (relative for same origin deployment)
const API_BASE = "";

// =============================================
// Drop Upload Setup
// =============================================

function setupDrop(dropId, inputId, previewId){

  const drop = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);

  const showPreview = (file)=>{
    const url = URL.createObjectURL(file);
    preview.src = url;
    drop.classList.add("has-file");
  };

  input.addEventListener("change",()=>{
    if(input.files[0]){
      showPreview(input.files[0]);
    }
  });

  drop.addEventListener("dragover", e=>{
    e.preventDefault();
    drop.classList.add("drag-over");
  });

  drop.addEventListener("dragleave", ()=>{
    drop.classList.remove("drag-over");
  });

  drop.addEventListener("drop", e=>{
    e.preventDefault();

    const file = e.dataTransfer.files[0];
    if(!file) return;

    const dt = new DataTransfer();
    dt.items.add(file);

    input.files = dt.files;
    showPreview(file);

    drop.classList.remove("drag-over");
  });
}

setupDrop("front-drop","front-input","front-preview");
setupDrop("back-drop","back-input","back-preview");
setupDrop("selfie-drop","selfie-input","selfie-preview");


// =============================================
// Document Preview Section
// =============================================

function showDocumentPreview(front,back,selfie){

  const section = document.getElementById("document-preview");
  if(!section) return;

  if(front){
    document.getElementById("preview-front").src =
      URL.createObjectURL(front);
  }

  if(back){
    document.getElementById("preview-back").src =
      URL.createObjectURL(back);
  }

  if(selfie){
    document.getElementById("preview-selfie").src =
      URL.createObjectURL(selfie);
  }

  section.style.display = "block";
}


// =============================================
// Progress Steps
// =============================================

const STEPS = [
  "load",
  "preprocess",
  "binarize",
  "ocr",
  "llm",
  "qr",
  "face",
  "verdict"
];

function activateStep(index){

  STEPS.forEach((s,i)=>{

    const el = document.querySelector(`[data-step="${s}"]`);

    if(!el) return;

    el.classList.remove("active","done","error");

    if(i < index) el.classList.add("done");
    else if(i === index) el.classList.add("active");

  });

}

function completeSteps(){

  STEPS.forEach(s=>{
    const el = document.querySelector(`[data-step="${s}"]`);
    if(el){
      el.classList.remove("active");
      el.classList.add("done");
    }
  });

  document.getElementById("main-spinner").style.display="none";
  document.getElementById("progress-title").textContent =
    "Verification complete";

}


// =============================================
// Form Submit
// =============================================

document.getElementById("upload-form")
.addEventListener("submit", async (e)=>{

  e.preventDefault();

  const frontFile = document.getElementById("front-input").files[0];
  const backFile = document.getElementById("back-input").files[0];
  const selfieFile = document.getElementById("selfie-input").files[0];

  if(!frontFile){
    showError("Please upload the front side.");
    return;
  }

  hideError();

  showDocumentPreview(frontFile,backFile,selfieFile);

  document.getElementById("progress-section").style.display="block";
  document.getElementById("verify-btn").disabled=true;

  activateStep(0);

  try{

    const formData = new FormData();

    formData.append("front",frontFile);
    if(backFile) formData.append("back",backFile);
    if(selfieFile) formData.append("selfie",selfieFile);

    const resp = await fetch(`${API_BASE}/api/validate`,{
      method:"POST",
      body:formData
    });

    if(!resp.ok){
      throw new Error("Server error");
    }

    const data = await resp.json();

    completeSteps();

    renderResult(data);

  }
  catch(err){

    showError("Verification failed: "+err.message);

  }

});


// =============================================
// Render Result
// =============================================

function renderResult(data){

  const {fields, verdict, qr, face} = data;

  document.getElementById("result-section").style.display="block";

  const score = verdict.score || 0;

  document.getElementById("score-value").textContent = score;

  const fill = document.getElementById("score-fill");
  const circumference = 188;

  fill.style.strokeDashoffset =
      circumference - (circumference*score/100);


  renderIdentity(fields);
  renderAddress(fields);

  renderFraudSignals(verdict);

  renderQR(qr);

  renderFace(face);

}


// =============================================
// Identity Table
// =============================================

function renderIdentity(fields){

  const table = document.getElementById("identity-table");

  const rows = [
    ["Name",fields.name],
    ["DOB",fields.dob],
    ["Gender",fields.gender],
    ["Aadhaar",fields.aadhaar_number],
    ["VID",fields.vid],
    ["Mobile",fields.mobile],
    ["Email",fields.email],
  ];

  table.innerHTML="";

  rows.forEach(r=>{
    table.innerHTML += `
    <tr>
      <td>${r[0]}</td>
      <td>${r[1] || "not found"}</td>
    </tr>
    `;
  });

}


// =============================================
// Address Table
// =============================================

function renderAddress(fields){

  const table = document.getElementById("address-table");

  const rows = [
    ["House",fields.address_house],
    ["Street",fields.address_street],
    ["Landmark",fields.address_landmark],
    ["District",fields.address_district],
    ["State",fields.address_state],
    ["PIN",fields.address_pin],
  ];

  table.innerHTML="";

  rows.forEach(r=>{
    table.innerHTML += `
    <tr>
      <td>${r[0]}</td>
      <td>${r[1] || "not found"}</td>
    </tr>
    `;
  });

}


// =============================================
// Fraud Signals
// =============================================

function renderFraudSignals(verdict){

  const fraudPanel = document.getElementById("fraud-panel");
  const fraudList = document.getElementById("fraud-list");

  fraudList.innerHTML="";

  if(verdict.issues && verdict.issues.length){

    fraudPanel.style.display="block";

    verdict.issues.forEach(issue=>{
      fraudList.innerHTML += `<div>• ${issue}</div>`;
    });

  } else {

    fraudPanel.style.display="none";

  }

}


// =============================================
// QR Metrics
// =============================================

function renderQR(qr){

  const trust = qr?.trust_score || 0;

  document.getElementById("qr-score").textContent = trust+"/100";

  const bar = document.getElementById("qr-bar");
  bar.style.width = trust+"%";

}


// =============================================
// Face Metrics
// =============================================

function renderFace(face){

  const score = face?.quality_score || 0;

  document.getElementById("face-score").textContent = score+"/100";

  const bar = document.getElementById("face-bar");
  bar.style.width = score+"%";

}


// =============================================
// Error Handling
// =============================================

function showError(msg){

  const b = document.getElementById("error-banner");

  b.textContent = "⚠ "+msg;
  b.style.display="block";

}

function hideError(){

  document.getElementById("error-banner").style.display="none";

}


// =============================================
// Reset Button
// =============================================

document.getElementById("reset-btn")
.addEventListener("click",()=>{

  document.getElementById("result-section").style.display="none";
  document.getElementById("progress-section").style.display="none";

  document.getElementById("upload-form").reset();

  ["front-drop","back-drop","selfie-drop"].forEach(id=>{
    document.getElementById(id).classList.remove("has-file");
  });

});