# Speech-to-Text Frontend

This project is a speech-to-text application that allows users to record audio, upload audio files, and view transcriptions of the audio. It is built using React and TypeScript, leveraging modern web technologies for a seamless user experience.

## Features

- **Audio Recording**: Users can record audio directly from their browser using the `AudioRecorder` component.
- **File Uploading**: Users can upload pre-recorded audio files for transcription using the `FileUploader` component.
- **Transcript Viewing**: The transcribed text is displayed in the `TranscriptViewer` component, allowing users to read and copy the text easily.
- **Responsive Design**: The application is designed to be responsive and user-friendly across different devices.

## Project Structure

```
speech-to-text-frontend
├── src
│   ├── components
│   │   ├── AudioRecorder.tsx
│   │   ├── FileUploader.tsx
│   │   ├── TranscriptViewer.tsx
│   │   └── Header.tsx
│   ├── pages
│   │   └── Home.tsx
│   ├── services
│   │   └── transcriptionApi.ts
│   ├── types
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── styles.css
├── public
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd speech-to-text-frontend
   ```
3. Install the dependencies:
   ```
   npm install
   ```

## Usage

To start the development server, run:
```
npm run dev
```

Open your browser and navigate to `http://localhost:3000` to view the application.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.