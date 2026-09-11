export interface TranscriptionResponse {
    text: string;
    language: string;
    filename: string;
}

const API_URL = '/api/transcribe';

const transcribe = async (
    audio: Blob,
    language: 'yo' | 'ha',
    filename: string,
): Promise<TranscriptionResponse> => {
    const formData = new FormData();
    formData.append('file', audio, filename);
    formData.append('language', language);

    const response = await fetch(API_URL, { method: 'POST', body: formData });
    if (!response.ok) {
        const body = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(body.detail || `Transcription failed (${response.status})`);
    }
    return response.json() as Promise<TranscriptionResponse>;
};

export const uploadAudioFile = (
    file: File,
    language: 'yo' | 'ha' = 'yo',
): Promise<TranscriptionResponse> => transcribe(file, language, file.name);

export const transcribeAudioData = (
    audioBlob: Blob,
    language: 'yo' | 'ha' = 'yo',
): Promise<TranscriptionResponse> => transcribe(audioBlob, language, 'recording.webm');