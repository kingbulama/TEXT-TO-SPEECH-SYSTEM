export interface TranscriptionResponse {
    text: string;
    confidence: number;
}

export interface AudioFile {
    name: string;
    size: number;
    type: string;
    lastModified: number;
}