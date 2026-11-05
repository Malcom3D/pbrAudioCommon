# pbrAudioCommon
Welcome to version v0.3.0 of pbrAudioCommon! This release brings major feature additions across our audio pipeline, including a brand new TUI for the DAW, new 3D viewport widgets and panel integration in Blender, advanced physical-to-acoustic processors, and major performance and storage enhancements.

* **New DAW Interface:** Experience the brand new `pbrAudioDAW` complete with a handy Terminal User Interface (TUI).
* **Blender Enhancements:** Enjoy a smoother workflow with new 3D viewport widgets and Viewport Panel integration in `pbrAudioBlender`.
* **Advanced Sound Processors:** New processors for modal displacement, audio forces, and mono modal displacement to acoustic pressure bring deeper physical modeling capabilities.
* **Storage & Performance:** `KleptoEntityStore` now supports bz2 compression, and `blosc2Backend` has been upgraded to parallel append and mmapped modes to handle large datasets seamlessly and prevent out-of-memory errors.
* **Bug Fixes:** Numerous typos, syntax errors, and data alignment issues between physics solvers and audio storage have been resolved for a much more stable experience.
