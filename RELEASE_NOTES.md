# pbrAudioCommon
This release introduces major updates to the pbrAudio ecosystem, focusing on performance, workflow integration, and stability.

**Key Updates:**

*   **New Tools & Integration:** We’ve added a new TUI-based DAW (`pbrAudioDAW`) and enhanced the Blender experience with new 3D viewport widgets and panels.
*   **Performance Improvements:** We have overhauled our storage backend to use parallel processing and memory-mapped files, which significantly improves data handling speed and stability. We also introduced bz2 compression to keep your project files efficient.
*   **Audio Processing:** New processors are now available for modal displacement and acoustic pressure, with improved support for complex audio signal chains.
*   **Bug Fixes:** This version includes critical fixes for memory management (addressing OOM issues in ModalComposer), metadata alignment, and various syntax/naming errors that could cause crashes.
*   **Workflow:** Data storage and metadata management are now more flexible, allowing for better customization of your audio processing chains.
