"""
Audio playback functionality for video viewing
"""

import os
import subprocess
import tempfile
from .constants import PYOPENXR_AVAILABLE

if PYOPENXR_AVAILABLE:
    import pygame


class AudioPlayer:
    """Manages audio extraction and playback for videos"""

    def __init__(self):
        self.initialized = False
        self.temp_file = None
        self.paused = False
        self.seek_offset = 0.0  # Track seek position offset

    def initialize(self):
        """Initialize pygame mixer"""
        if not PYOPENXR_AVAILABLE:
            return False

        if not self.initialized:
            try:
                pygame.mixer.init()
                self.initialized = True
                return True
            except Exception as e:
                print(f"   ⚠️  Audio initialization failed: {e}")
                return False
        return True

    def detect_audio_codec(self, video_path):
        """
        Detect the audio codec of the video file

        Args:
            video_path: Path to video file

        Returns:
            str: Audio codec name, or None if detection fails
        """
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout.strip():
                codec = result.stdout.strip()
                return codec
            return None

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def load_from_video(self, video_path, loop=True):
        """
        Extract and load audio from video file using pygame
        Uses fast codec copy when possible, falls back to re-encoding if needed

        Args:
            video_path: Path to video file
            loop: Whether to loop audio playback

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialize():
            return False

        try:
            # Clean up previous audio temp file
            self.cleanup_temp_file()

            # Detect audio codec
            audio_codec = self.detect_audio_codec(video_path)
            print(f"   Detected audio codec: {audio_codec if audio_codec else 'unknown'}")

            # Determine if we can use fast copy or need re-encoding
            # pygame.mixer supports: MP3, OGG, WAV, FLAC, MOD
            fast_copy_codecs = ['mp3', 'vorbis', 'opus', 'flac', 'pcm_s16le', 'pcm_s24le', 'pcm_s32le']
            use_fast_copy = audio_codec in fast_copy_codecs

            # Choose appropriate file extension based on codec
            if use_fast_copy:
                if audio_codec in ['vorbis', 'opus']:
                    suffix = '.ogg'
                elif audio_codec == 'flac':
                    suffix = '.flac'
                elif audio_codec.startswith('pcm'):
                    suffix = '.wav'
                else:  # mp3
                    suffix = '.mp3'
            else:
                suffix = '.ogg'  # Use OGG for re-encoding (better quality than MP3)

            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            self.temp_file = temp_audio.name
            temp_audio.close()

            # Try to extract audio using ffmpeg (if available)
            try:
                if use_fast_copy:
                    # Fast path: copy audio without re-encoding
                    print(f"   Using fast copy (no re-encoding)...")
                    result = subprocess.run([
                        'ffmpeg', '-i', video_path,
                        '-vn',  # No video
                        '-acodec', 'copy',  # Copy audio without re-encoding
                        '-y',  # Overwrite
                        self.temp_file
                    ], capture_output=True, check=True, timeout=30)
                else:
                    # Slow path: re-encode to OGG Vorbis
                    print(f"   Re-encoding audio to OGG Vorbis...")
                    result = subprocess.run([
                        'ffmpeg', '-i', video_path,
                        '-vn',  # No video
                        '-acodec', 'libvorbis',  # OGG Vorbis codec
                        '-q:a', '6',  # Quality level 6 (good balance)
                        '-y',  # Overwrite
                        self.temp_file
                    ], capture_output=True, check=True, timeout=30)

                # Load audio with pygame
                pygame.mixer.music.load(self.temp_file)
                pygame.mixer.music.play(loops=-1 if loop else 0)
                self.seek_offset = 0.0  # Reset seek offset on new load
                print("   ✓ Audio loaded and playing")
                return True

            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                print("   ⚠️  Could not extract audio (ffmpeg not found or no audio track)")
                self.cleanup_temp_file()
                return False

        except Exception as e:
            print(f"   ⚠️  Audio loading failed: {e}")
            self.cleanup_temp_file()
            return False

    def pause(self):
        """Pause audio playback"""
        if self.initialized and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True

    def unpause(self):
        """Resume audio playback"""
        if self.initialized and self.paused:
            pygame.mixer.music.unpause()
            self.paused = False

    def toggle(self):
        """Toggle audio play/pause"""
        if self.paused:
            self.unpause()
        else:
            self.pause()

    def stop(self):
        """Stop audio playback"""
        if self.initialized:
            pygame.mixer.music.stop()
        self.cleanup_temp_file()

    def restart(self):
        """Restart audio from beginning"""
        if self.initialized:
            pygame.mixer.music.rewind()
            self.seek_offset = 0.0
            if not self.paused:
                pygame.mixer.music.unpause()

    def seek(self, position_seconds):
        """
        Seek audio to a specific position in seconds.

        Args:
            position_seconds: Target position in seconds
        """
        if self.initialized:
            try:
                position_seconds = max(0.0, position_seconds)

                # pygame's set_pos behavior is inconsistent across formats
                # More reliable approach: stop, then play from position
                # pygame.mixer.music.play(start=position) starts from that position
                was_paused = self.paused

                # Restart playback from the new position
                # The start parameter specifies where to begin (in seconds)
                pygame.mixer.music.play(loops=-1, start=position_seconds)
                self.seek_offset = position_seconds

                if was_paused:
                    pygame.mixer.music.pause()

                print(f"   🔊 Audio seeked to {position_seconds:.1f}s")
            except Exception as e:
                print(f"   ⚠️  Audio seek failed: {e}")

    def seek_relative(self, offset_seconds):
        """
        Seek audio by a relative offset in seconds.

        Args:
            offset_seconds: Offset in seconds (positive = forward, negative = backward)
        """
        current_pos = self.get_position()
        if current_pos is not None:
            new_pos = current_pos + offset_seconds
            self.seek(new_pos)

    def get_position(self):
        """
        Get current playback position in seconds.
        Returns None if audio is not playing or position unavailable.
        """
        if self.initialized and not self.paused:
            try:
                # pygame.mixer.music.get_pos() returns milliseconds since last play/seek
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    return self.seek_offset + (pos_ms / 1000.0)
            except Exception:
                pass
        return None

    def is_playing(self):
        """Check if audio is currently playing"""
        if self.initialized:
            return pygame.mixer.music.get_busy() and not self.paused
        return False

    def cleanup_temp_file(self):
        """Clean up temporary audio file"""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except Exception:
                pass  # Ignore cleanup errors
            self.temp_file = None

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
