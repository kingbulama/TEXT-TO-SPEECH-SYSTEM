import React from 'react';

interface TranscriptViewerProps {
    transcript: string;
}

const TranscriptViewer: React.FC<TranscriptViewerProps> = ({ transcript }) => {
    return (
        <div className="transcript-viewer">
            <h2>Transcribed Text</h2>
            <p>{transcript}</p>
        </div>
    );
};

export default TranscriptViewer;