// verification.model.ts
// TypeScript interfaces matching the AadhaarCheck v5 backend JSON response

export interface VerificationFields {
  name:                string | null;
  name_hindi:          string | null;
  dob:                 string | null;
  year_of_birth:       string | null;
  gender:              string | null;
  aadhaar_number:      string | null;
  vid:                 string | null;
  enrollment_number:   string | null;
  father_husband_name: string | null;
  mobile:              string | null;
  email:               string | null;
  address_raw:         string | null;
  address_house:       string | null;
  address_street:      string | null;
  address_landmark:    string | null;
  address_locality:    string | null;
  address_subdistrict: string | null;
  address_district:    string | null;
  address_state:       string | null;
  address_pin:         string | null;
  issue_date:          string | null;
  card_type:           string | null;
  issued_by:           string | null;
}

export interface VerificationVerdict {
  label:   string;
  detail:  string;
  score:   number;
  color:   'green' | 'yellow' | 'orange' | 'red';
  passed:  string[];
  issues:  string[];
}

export interface QRFieldCheck {
  match: boolean | null;
  note:  string;
}

export interface QRFields {
  name:          string | null;
  dob:           string | null;
  gender:        string | null;
  pin:           string | null;
  district:      string | null;
  state:         string | null;
  building:      string | null;
  street:        string | null;
  landmark:      string | null;
  locality:      string | null;
  vtc:           string | null;
  mobile_masked: string | null;
  mobile_last4:  string | null;
  reference_id:  string | null;
  qr_version:    string | null;
}

export interface QRResult {
  qr_found:         boolean;
  qr_decoded:       boolean;
  qr_format:        string | null;
  trust_score:      number;
  verdict:          string;
  qr_fields:        QRFields;
  field_checks:     Record<string, QRFieldCheck>;
  fraud_signals:    string[];
  qr_detect_source: string | null;
}

export interface FaceAgeDetail {
  estimated_age: number;
  declared_age:  number;
  age_diff:      number;
  method:        string;
}

export interface FaceMatchResult {
  match:       boolean;
  match_score: number;
  engine:      string;
  verdict:     string;
  note:        string;
}

export interface FaceResult {
  quality_score:            number;
  quality_verdict:          string;
  liveness_score:           number | null;
  likely_real:              boolean | null;
  combined_liveness_score:  number | null;
  combined_liveness_real:   boolean | null;
  passive_liveness_score:   number | null;
  age_score:                number | null;
  age_note:                 string | null;
  age_detail:               FaceAgeDetail | null;
  occlusion_score:          number | null;
  occlusion_note:           string | null;
  extraction_method:        string | null;
  selfie_provided:          boolean;
  match_result:             FaceMatchResult | null;
}

export interface DetectorDetail {
  score:  number;
  note:   string;
  passed: boolean;
}

export interface TamperResult {
  score:   number;
  verdict: string;
  signals: string[];
  passed:  boolean;
  details: {
    ela:      DetectorDetail;
    noise:    DetectorDetail;
    font:     DetectorDetail;
    moire:    DetectorDetail;
    hologram: DetectorDetail;
    verhoeff: DetectorDetail;
  };
}

export interface GeoResult {
  score:   number;
  verdict: string;
  signals: string[];
  passed:  boolean;
  details: {
    pin_geo:           DetectorDetail;
    district_state:    DetectorDetail;
    ai_image:          DetectorDetail;
    name_plausibility: DetectorDetail;
  };
}

export interface Explanation {
  source:   string;
  signal:   string;
  plain:    string;
  severity: 'critical' | 'warning' | 'info';
}

export interface VerificationResponse {
  fields:           VerificationFields;
  verdict:          VerificationVerdict;
  qr:               QRResult;
  face:             FaceResult;
  tamper:           TamperResult;
  geo:              GeoResult;
  explanations:     Explanation[];
  pdf_report_id:    string | null;
  pdf_report_url:   string | null;
  pdf_report_hash:  string | null;
}

export type VerificationStep =
  | 'idle'
  | 'loading'
  | 'preprocess'
  | 'binarize'
  | 'ocr'
  | 'llm'
  | 'qr'
  | 'face'
  | 'verdict'
  | 'done'
  | 'error';

export interface PipelineStep {
  id:    VerificationStep;
  label: string;
  icon:  string;
}

export const PIPELINE_STEPS: PipelineStep[] = [
  { id: 'loading',    label: 'Loading images',                  icon: '📥' },
  { id: 'preprocess', label: 'Preprocessing — resize · CLAHE',  icon: '🔧' },
  { id: 'binarize',   label: 'Binarize · deskew · morphology',  icon: '⬛' },
  { id: 'ocr',        label: 'OCR — PaddleOCR + Tesseract',     icon: '🔍' },
  { id: 'llm',        label: 'LLM correction · field extract',  icon: '🧠' },
  { id: 'qr',         label: 'QR Secure decode · cross-check',  icon: '📱' },
  { id: 'face',       label: 'Face AI · liveness · age · match',icon: '👁️' },
  { id: 'verdict',    label: 'Tampering · geo · final verdict', icon: '⚖️' },
];
