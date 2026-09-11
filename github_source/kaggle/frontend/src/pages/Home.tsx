import React, { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react';

type Language = 'yo' | 'ha';

const languageNames: Record<Language, string> = {
    yo: 'Yoruba',
    ha: 'Hausa',
};

const Home: React.FC = () => {
    const [language, setLanguage] = useState<Language>('yo');
    const [file, setFile] = useState<File | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [error, setError] = useState('');
    const [showSettings, setShowSettings] = useState(false);
    const [showAllSessions, setShowAllSessions] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const recorderRef = useRef<MediaRecorder | null>(null);
    const recordingChunksRef = useRef<Blob[]>([]);

    useEffect(() => {
        return () => {
            if (audioUrl) URL.revokeObjectURL(audioUrl);
        };
    }, [audioUrl]);

    const selectFile = (nextFile: File | undefined) => {
        if (!nextFile) return;
        if (!nextFile.type.startsWith('audio/')) {
            setError('Choose an audio file such as WAV, MP3, M4A, or OGG.');
            return;
        }
        setFile(nextFile);
        setAudioUrl(URL.createObjectURL(nextFile));
        setTranscript('');
        setError('');
    };

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        selectFile(event.target.files?.[0]);
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        selectFile(event.dataTransfer.files[0]);
    };

    const handleDropzoneKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            inputRef.current?.click();
        }
    };

    const toggleRecording = async () => {
        if (isRecording && recorderRef.current) {
            recorderRef.current.stop();
            setIsRecording(false);
            return;
        }
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
            setError('Audio recording is not supported in this browser. Upload a file instead.');
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            recordingChunksRef.current = [];
            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) recordingChunksRef.current.push(event.data);
            };
            recorder.onstop = () => {
                stream.getTracks().forEach((track) => track.stop());
                const recordedFile = new File(
                    recordingChunksRef.current,
                    `recording-${new Date().toISOString().slice(0, 10)}.webm`,
                    { type: recorder.mimeType || 'audio/webm' },
                );
                selectFile(recordedFile);
                recorderRef.current = null;
            };
            recorderRef.current = recorder;
            recorder.start();
            setError('');
            setIsRecording(true);
        } catch {
            setError('Microphone access was blocked. Allow microphone access or upload an audio file.');
        }
    };

    const startNewSession = () => {
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setFile(null);
        setAudioUrl(null);
        setTranscript('');
        setError('');
        setIsRecording(false);
        if (inputRef.current) inputRef.current.value = '';
    };

    const copyTranscript = async () => {
        if (!transcript) return;
        try {
            await navigator.clipboard.writeText(transcript);
            setError('Transcript copied to your clipboard.');
        } catch {
            setError('Clipboard access is unavailable. Select the transcript and copy it manually.');
        }
    };

    const transcribe = async () => {
        if (!file) {
            setError('Add an audio file before starting transcription.');
            return;
        }
        setIsTranscribing(true);
        setError('');
        try {
            const body = new FormData();
            body.append('file', file);
            body.append('language', language);
            const response = await fetch('/api/transcribe', { method: 'POST', body });
            if (!response.ok) throw new Error('API unavailable');
            const data = await response.json() as { text?: string; transcript?: string };
            setTranscript(data.text || data.transcript || 'No transcript returned.');
        } catch (requestError) {
            const message = requestError instanceof Error ? requestError.message : 'Transcription request failed.';
            setError(`Transcription failed: ${message}. Start the local API and check the model path.`);
        } finally {
            setIsTranscribing(false);
        }
    };

    return (
        <div className="app-shell">
            <header className="topbar">
                <button className="brand-lockup brand-button" onClick={startNewSession} aria-label="Start a new session"><span className="brand-mark">/</span><span>VOXARA</span></button>
                <div className="topbar-meta"><span className="live-dot" /> Local workspace <span className="divider" /> Model: Whisper LoRA</div>
                <button className="icon-button" aria-label="Open settings" onClick={() => setShowSettings(true)}>•••</button>
            </header>

            <main className="workspace">
                <section className="intro-row">
                    <div>
                        <p className="eyebrow">MULTILINGUAL ASR / WORKSPACE 01</p>
                        <h1>Turn voice into <em>meaning.</em></h1>
                        <p className="intro-copy">Transcribe Yoruba and Hausa audio with your fine-tuned multilingual model.</p>
                    </div>
                    <div className="session-count"><span>SESSION</span><strong>004</strong><small>Ready to process</small></div>
                </section>

                <div className="main-grid">
                    <section className="control-panel">
                        <div className="section-heading"><div><span className="step-number">01</span><h2>Choose language</h2></div><span className="status-pill">MODEL READY</span></div>
                        <div className="language-switcher">
                            {(Object.keys(languageNames) as Language[]).map((code) => (
                                <button key={code} className={language === code ? 'language-option selected' : 'language-option'} onClick={() => setLanguage(code)}>
                                    <span className="language-code">{code}</span><span>{languageNames[code]}</span><span className="radio-dot" />
                                </button>
                            ))}
                        </div>

                        <div className="section-heading upload-heading"><div><span className="step-number">02</span><h2>Add audio</h2></div><span className="file-types">WAV · MP3 · M4A</span></div>
                        <div className={file ? 'dropzone has-file' : 'dropzone'} onDrop={handleDrop} onDragOver={(event) => event.preventDefault()} onClick={() => inputRef.current?.click()} onKeyDown={handleDropzoneKeyDown} role="button" tabIndex={0}>
                            <input ref={inputRef} type="file" accept="audio/*" onChange={handleFileChange} hidden />
                            {file ? <><span className="upload-icon">✓</span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} MB · Ready to transcribe</small></> : <><span className="upload-icon">↑</span><strong>Drop your audio here</strong><small>or click to browse your files</small></>}
                        </div>
                        {error && <p className="helper-error">{error}</p>}
                        <div className="action-row"><button className="record-button" onClick={toggleRecording}><span className={isRecording ? 'record-dot recording' : 'record-dot'} />{isRecording ? 'Stop recording' : 'Record audio'}</button><button className="primary-button" disabled={!file || isTranscribing} onClick={transcribe}>{isTranscribing ? 'Transcribing…' : 'Start transcription'}<span>→</span></button></div>
                    </section>

                    <section className="transcript-panel">
                        <div className="section-heading"><div><span className="step-number">03</span><h2>Transcript</h2></div><button className="copy-button" disabled={!transcript} onClick={copyTranscript}>Copy text</button></div>
                        {audioUrl && <div className="audio-preview"><audio controls src={audioUrl} /><div className="waveform" aria-hidden="true">{Array.from({ length: 48 }, (_, index) => <i key={index} style={{ height: `${18 + ((index * 17) % 38)}%` }} />)}</div></div>}
                        <div className={transcript ? 'transcript-output filled' : 'transcript-output'}>{transcript ? <p>{transcript}</p> : <><span className="empty-orbit">◎</span><p>Your transcript will appear here</p><small>Upload an audio file and start a transcription</small></>}</div>
                        <div className="metric-row"><div><span>LANGUAGE</span><strong>{languageNames[language]}</strong></div><div><span>CONFIDENCE</span><strong>{transcript ? '—' : 'Awaiting'}</strong></div><div><span>WORDS</span><strong>{transcript ? transcript.trim().split(/\s+/).length : '—'}</strong></div></div>
                    </section>
                </div>

                <section className="recent-section"><div className="recent-title"><p className="eyebrow">RECENT SESSIONS</p><button className="text-button" onClick={() => setShowAllSessions(!showAllSessions)}>{showAllSessions ? 'Hide' : 'View all'} <span>→</span></button></div><div className="recent-grid"><article><div className="session-icon yo">YO</div><div><strong>morning-thoughts.wav</strong><small>Yoruba · 00:42</small></div><span className="session-status">Completed</span><span className="chevron">→</span></article><article><div className="session-icon ha">HA</div><div><strong>field-interview-02.mp3</strong><small>Hausa · 01:18</small></div><span className="session-status">Completed</span><span className="chevron">→</span></article>{showAllSessions && <article><div className="session-icon yo">YO</div><div><strong>community-check.wav</strong><small>Yoruba · 00:36</small></div><span className="session-status">Completed</span><span className="chevron">→</span></article>}<button className="empty-session" onClick={startNewSession}><span>+</span><strong>New session</strong></button></div></section>
            </main>
            <footer><span>VOXARA ASR</span><span>Built for Yoruba + Hausa</span><span>CPU inference available</span></footer>
            {showSettings && <div className="modal-backdrop" role="presentation" onClick={() => setShowSettings(false)}><section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title" onClick={(event) => event.stopPropagation()}><div className="modal-heading"><div><p className="eyebrow">WORKSPACE SETTINGS</p><h2 id="settings-title">Transcription setup</h2></div><button className="close-button" onClick={() => setShowSettings(false)} aria-label="Close settings">×</button></div><label>Model endpoint<input defaultValue="/api/transcribe" readOnly /></label><label>Inference device<select defaultValue="CPU"><option>CPU</option><option>CUDA (when available)</option></select></label><button className="primary-button modal-save" onClick={() => setShowSettings(false)}>Done <span>→</span></button></section></div>}
        </div>
    );
};

export default Home;