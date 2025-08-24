// content.js - Advanced YouTube Shorts AI Personalizer (User Intent Detection Version)

// Configuration - ADJUST THESE VALUES TO CUSTOMIZE BEHAVIOR
const CONFIG = {
  autoScroll: true,
  autoFeedback: true,
  zeroDelayScroll: true,
  autoSkipThreshold: 0,
  likeThreshold: 2,           // LOWER = MORE SENSITIVE (try 1-2 for very sensitive, 5-10 for less sensitive)
  dislikeThreshold: -1,       // LOWER = MORE SENSITIVE (try -1 for very sensitive, -5 for less sensitive)
  minScrollInterval: 500,     // Reduced for better responsiveness
  debugMode: true,
  statusOverlay: true,
  overlayPosition: 'bottom-left',
  apiBaseUrl: 'http://localhost:8000',
  bufferSize: 20,
  checkVideoInterval: 25,
  trustButtonUpdateInterval: 400,
  videoCompletionThreshold: 85, // Completion percentage threshold
  channelDetectionAttempts: 15,
  forceScrollAfterSeconds: 13131313131313131313,
  aggressiveChannelDetection: true,
  showBufferInOverlay: true,
  watchedPercentUpdateInterval: 150,
  feedbackWatchThreshold: 15, // Apply auto-feedback after watching this percentage
  badVideoScrollDelay: 200,
  moods: ["Neutral", "Happy", "Relaxed", "Focused", "Energetic", "Curious", "Creative", "Mad"],
  autoScrollOnCompletion: true,
  autoFeedbackConfirmationTime: 30000, // 30 seconds to undo auto-feedback before AI learns from it
  longFormClickPreventionDelay: 300, // Prevent accidental clicks for 300ms after page load
  userIntentDetectionTime: 5000, // 5 seconds to detect user intent to stay on video
  userIntentScrollBackThreshold: 2 // Number of scroll-backs to trigger intent detection
};

// State
let STATE = {
  currentVideoId: null,
  currentChannelId: null,
  currentTitle: null,
  currentDescription: null,
  currentCaptions: null,
  lastScrollTime: 0,
  watchStartTime: 0,
  watchedPercent: 0,
  lastValidWatchedPercent: 0,
  maxWatchedPercent: 0,
  videoCompleted: false,
  videoDisliked: false,
  pendingAutoFeedback: false,
  autoFeedbackType: null,
  autoFeedbackTimestamp: 0, // When auto-feedback was given
  autoFeedbackConfirmed: false, // Whether auto-feedback was confirmed (not undone )
  metadataExtracted: false,
  score: 0,
  statusElement: null,
  controlPanel: null,
  controlPanelVisible: false,
  trustButton: null,
  trustButtonVisible: false,
  lastVideoChange: 0,
  correctionsMade: 0,
  autoScrollEnabled: CONFIG.autoScroll,
  autoFeedbackEnabled: CONFIG.autoFeedback,
  channelIsTrusted: false,
  channelIsBlocked: false,
  lastChannelStatusCheck: 0,
  videoElement: null,
  videoChangeDetected: false,
  scrollAttempts: 0,
  lastScrollAttempt: 0,
  scrollMethods: ['keydown', 'click', 'scroll', 'swipe', 'api'],
  currentScrollMethod: 0,
  videoCompletionDetectionMethods: ['ended', 'percent', 'stalled', 'timeupdate'],
  completionDetectionMethod: null,
  channelDetectionAttempts: 0,
  lastChannelDetectionAttempt: 0,
  watchTimeStart: 0,
  bufferSize: CONFIG.bufferSize,
  lastWatchedPercentUpdate: 0,
  forceScrollTimer: null,
  lastVideoElementCheck: 0,
  lastTrustButtonUpdate: 0,
  lastAutoScrollTrigger: null,
  badVideoCount: 0,
  lastBadVideoTimestamp: 0,
  lastAutoScrolledVideoId: null,
  currentMood: 'Neutral',
  userHasGivenFeedback: false, // Track if user has manually given feedback
  aiActionBlocked: false, // Block AI actions when user has taken control
  completionCheckInterval: null, // Interval for checking completion
  autoFeedbackConfirmationTimer: null, // Timer for confirming auto-feedback
  pageLoadTime: Date.now(), // Track when page loaded for click prevention
  stuckPercentageCount: 0, // Track how long percentage has been stuck
  lastPercentageValue: 0, // Track last percentage value
  
  // NEW: User intent detection
  userIntentToStay: false, // Whether user has shown intent to stay on current video
  videoReturnCount: 0, // How many times user has returned to this video
  lastScrollDirection: null, // 'up' or 'down'
  videoStayTimer: null, // Timer to detect if user stays on video
  userIntentDetected: false, // Whether we've detected user intent for current video
  lastVideoInHistory: [], // Track last few videos to detect returns
  scrollBackCount: 0, // Count of consecutive scroll-backs to same video
  userOverrideActive: false // Whether user has overridden AI for current video
};

console.log("[ShortsAI] Content script loading (User Intent Detection Version)...");
console.log("[ShortsAI] Like threshold:", CONFIG.likeThreshold, "| Dislike threshold:", CONFIG.dislikeThreshold);

// --- Core Initialization and DOM Monitoring ---

const observer = new MutationObserver(debounce(handleDOMChanges, 500));

document.addEventListener('DOMContentLoaded', function() {
  console.log('[ShortsAI] DOM fully loaded, initializing...');
  setTimeout(initialize, 1000);
});

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  console.log('[ShortsAI] Document already loaded, initializing...');
  setTimeout(initialize, 1000);
}

function initialize() {
  console.log("[ShortsAI] Initializing...");
  try {
    createStatusOverlay();
    createTrustButton();
    setupKeyboardShortcuts();

    // Smart long-form video click prevention
    document.addEventListener("click", (event) => {
      const longFormLink = event.target.closest('a[href*="/watch?"]');
      if (longFormLink) {
        const timeSincePageLoad = Date.now() - STATE.pageLoadTime;
        const timeSinceVideoChange = Date.now() - STATE.lastVideoChange;
        
        if (timeSincePageLoad < CONFIG.longFormClickPreventionDelay || 
            timeSinceVideoChange < CONFIG.longFormClickPreventionDelay) {
          console.log("[ShortsAI] Prevented likely accidental click on long-form video link");
          event.preventDefault();
          event.stopPropagation();
          return;
        } else {
          console.log("[ShortsAI] Allowing intentional long-form video click");
        }
      }

      // Detect manual like/dislike actions
      const likeButton = event.target.closest('yt-icon-button[aria-label*="like"], button[aria-label*="like"]');
      const dislikeButton = event.target.closest('yt-icon-button[aria-label*="dislike"], button[aria-label*="dislike"]');

      if (likeButton && STATE.currentVideoId) {
        console.log("[ShortsAI] Manual like action detected - USER INTENT TO STAY");
        handleUserIntentToStay("manual_like");
        STATE.userHasGivenFeedback = true;
        STATE.aiActionBlocked = true;
        
        if (STATE.autoFeedbackConfirmationTimer) {
          clearTimeout(STATE.autoFeedbackConfirmationTimer);
          STATE.autoFeedbackConfirmationTimer = null;
        }
        
        if (STATE.autoFeedbackType === "dislike" && !STATE.autoFeedbackConfirmed) {
          console.log("[ShortsAI] User corrected auto-dislike with manual like - PENALTY TO AI");
          sendEvent("undo_auto_dislike", -12);
        } else if (STATE.autoFeedbackType === "like" && !STATE.autoFeedbackConfirmed) {
          console.log("[ShortsAI] User undid auto-like - PENALTY TO AI");
          sendEvent("undo_auto_like", -10);
        }
        
        setTimeout(() => {
          const isNowLiked = likeButton.getAttribute('aria-pressed') === 'true' || 
                           likeButton.classList.contains('style-default-active');
          
          if (isNowLiked) {
            console.log("[ShortsAI] Video manually liked by user - POSITIVE FEEDBACK");
            sendEvent("user_like", 12);
          } else {
            console.log("[ShortsAI] User removed like");
            sendEvent("user_unlike", 0);
          }
        }, 300);
        
      } else if (dislikeButton && STATE.currentVideoId) {
        console.log("[ShortsAI] Manual dislike action detected - BLOCKING ALL AI ACTIONS");
        STATE.userHasGivenFeedback = true;
        STATE.aiActionBlocked = true;
        
        if (STATE.autoFeedbackConfirmationTimer) {
          clearTimeout(STATE.autoFeedbackConfirmationTimer);
          STATE.autoFeedbackConfirmationTimer = null;
        }
        
        if (STATE.autoFeedbackType === "like" && !STATE.autoFeedbackConfirmed) {
          console.log("[ShortsAI] User corrected auto-like with manual dislike - PENALTY TO AI");
          sendEvent("undo_auto_like", -12);
        } else if (STATE.autoFeedbackType === "dislike" && !STATE.autoFeedbackConfirmed) {
          console.log("[ShortsAI] User undid auto-dislike - PENALTY TO AI");
          sendEvent("undo_auto_dislike", -10);
        }
        
        setTimeout(() => {
          const isNowDisliked = dislikeButton.getAttribute('aria-pressed') === 'true' || 
                               dislikeButton.classList.contains('style-default-active');
          
          if (isNowDisliked) {
            console.log("[ShortsAI] Video manually disliked by user - NEGATIVE FEEDBACK");
            sendEvent("user_dislike", -12);
          } else {
            console.log("[ShortsAI] User removed dislike");
            sendEvent("user_undislike", 0);
          }
        }, 300);
      }
    }, true);

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // NEW: Enhanced scroll detection for user intent
    document.addEventListener("scroll", debounce((event) => {
      handleScrollEvent(event);
    }, 100));

    handleDOMChanges();
    
    setInterval(checkVideoState, CONFIG.checkVideoInterval);
    setInterval(updateTrustButton, CONFIG.trustButtonUpdateInterval);
    setInterval(updateWatchedPercent, CONFIG.watchedPercentUpdateInterval);
    
    fetchBufferSize();

    console.log("[ShortsAI] Initialization complete. Monitoring DOM for changes.");
  } catch (error) {
    console.error("[ShortsAI] Initialization failed:", error);
  }
}

// NEW: Enhanced scroll event handler with user intent detection
function handleScrollEvent(event) {
  try {
    const currentScrollY = window.scrollY;
    const previousScrollY = STATE.lastScrollY || 0;
    
    // Determine scroll direction
    if (currentScrollY > previousScrollY) {
      STATE.lastScrollDirection = 'down';
    } else if (currentScrollY < previousScrollY) {
      STATE.lastScrollDirection = 'up';
      
      // User scrolled up - check if they're returning to a video
      checkForVideoReturn();
    }
    
    STATE.lastScrollY = currentScrollY;
    
    // Handle manual scroll (existing functionality)
    if (STATE.currentVideoId && !STATE.videoCompleted) {
      console.log('[ShortsAI] Manual scroll detected');
      sendEvent('manual_skip', STATE.watchedPercent);
    }
  } catch (error) {
    console.error('[ShortsAI] Error in handleScrollEvent:', error);
  }
}

// NEW: Check if user is returning to a previous video
function checkForVideoReturn() {
  try {
    if (!STATE.currentVideoId) return;
    
    // Check if current video was in recent history (user returned to it)
    const wasInHistory = STATE.lastVideoInHistory.includes(STATE.currentVideoId);
    
    if (wasInHistory) {
      STATE.scrollBackCount++;
      console.log(`[ShortsAI] User returned to video ${STATE.currentVideoId} (return #${STATE.scrollBackCount})`);
      
      // If user has returned multiple times, they clearly want to watch this
      if (STATE.scrollBackCount >= CONFIG.userIntentScrollBackThreshold) {
        console.log("[ShortsAI] 🎯 USER INTENT DETECTED: Multiple returns to same video!");
        handleUserIntentToStay("multiple_returns");
      } else {
        // Start timer to see if they stay
        startUserIntentTimer();
      }
    }
  } catch (error) {
    console.error('[ShortsAI] Error in checkForVideoReturn:', error);
  }
}

// NEW: Start timer to detect if user stays on video
function startUserIntentTimer() {
  try {
    // Clear existing timer
    if (STATE.videoStayTimer) {
      clearTimeout(STATE.videoStayTimer);
    }
    
    console.log(`[ShortsAI] Starting user intent timer (${CONFIG.userIntentDetectionTime/1000}s)`);
    
    STATE.videoStayTimer = setTimeout(() => {
      if (STATE.currentVideoId && !STATE.userIntentDetected) {
        console.log("[ShortsAI] 🎯 USER INTENT DETECTED: Stayed on video for 5+ seconds!");
        handleUserIntentToStay("stayed_on_video");
      }
    }, CONFIG.userIntentDetectionTime);
  } catch (error) {
    console.error('[ShortsAI] Error in startUserIntentTimer:', error);
  }
}

// NEW: Handle when user shows intent to stay on current video
function handleUserIntentToStay(reason) {
  try {
    if (STATE.userIntentDetected) return; // Already detected for this video
    
    console.log(`[ShortsAI] 🎯 USER INTENT TO STAY DETECTED: ${reason}`);
    console.log("[ShortsAI] *SLAPS ALGORITHM* User wants to watch this video!");
    
    STATE.userIntentToStay = true;
    STATE.userIntentDetected = true;
    STATE.userOverrideActive = true;
    STATE.aiActionBlocked = true; // Block all AI actions
    
    // Clear any pending auto-scroll or auto-feedback
    if (STATE.forceScrollTimer) {
      clearTimeout(STATE.forceScrollTimer);
      STATE.forceScrollTimer = null;
      console.log("[ShortsAI] Cancelled force scroll timer - user wants to stay");
    }
    
    if (STATE.autoFeedbackConfirmationTimer) {
      clearTimeout(STATE.autoFeedbackConfirmationTimer);
      STATE.autoFeedbackConfirmationTimer = null;
      console.log("[ShortsAI] Cancelled auto-feedback timer - user override active");
    }
    
    // Send strong positive signal to AI
    sendEvent("user_intent_to_stay", 15); // Strong positive signal
    
    // Update status overlay
    updateStatusOverlay();
    
    console.log("[ShortsAI] AI actions blocked - user is in control of this video");
  } catch (error) {
    console.error('[ShortsAI] Error in handleUserIntentToStay:', error);
  }
}

function debounce(func, delay) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), delay);
  };
}

function handleDOMChanges() {
  try {
    detectCurrentVideo();
  } catch (error) {
    console.error("[ShortsAI] Error handling DOM changes:", error);
  }
}

// --- Video and Metadata Detection ---

function detectCurrentVideo() {
  try {
    if (!window.location.href.includes('/shorts/')) {
      return;
    }
    
    const urlMatch = window.location.href.match(/\/shorts\/([^/?]+)/);
    if (!urlMatch) {
      return;
    }
    
    const videoId = urlMatch[1];
    
    if (videoId !== STATE.currentVideoId) {
      console.log(`[ShortsAI] Video changed: ${videoId}`);
      
      // NEW: Add previous video to history
      if (STATE.currentVideoId) {
        STATE.lastVideoInHistory.unshift(STATE.currentVideoId);
        if (STATE.lastVideoInHistory.length > 5) {
          STATE.lastVideoInHistory.pop(); // Keep only last 5 videos
        }
      }
      
      // Update page load time for new video
      STATE.pageLoadTime = Date.now();
      
      // Clear all timers
      if (STATE.forceScrollTimer) {
        clearTimeout(STATE.forceScrollTimer);
        STATE.forceScrollTimer = null;
      }
      if (STATE.completionCheckInterval) {
        clearInterval(STATE.completionCheckInterval);
        STATE.completionCheckInterval = null;
      }
      if (STATE.autoFeedbackConfirmationTimer) {
        clearTimeout(STATE.autoFeedbackConfirmationTimer);
        STATE.autoFeedbackConfirmationTimer = null;
      }
      if (STATE.videoStayTimer) {
        clearTimeout(STATE.videoStayTimer);
        STATE.videoStayTimer = null;
      }
      
      // Reset state for new video
      STATE.currentVideoId = videoId;
      STATE.currentChannelId = null;
      STATE.currentTitle = null;
      STATE.currentDescription = null;
      STATE.currentCaptions = null;
      STATE.watchedPercent = 0;
      STATE.lastValidWatchedPercent = 0;
      STATE.maxWatchedPercent = 0;
      STATE.videoCompleted = false;
      STATE.videoDisliked = false;
      STATE.pendingAutoFeedback = false;
      STATE.autoFeedbackType = null;
      STATE.autoFeedbackTimestamp = 0;
      STATE.autoFeedbackConfirmed = false;
      STATE.metadataExtracted = false;
      STATE.score = 0;
      STATE.lastVideoChange = Date.now();
      STATE.watchStartTime = Date.now();
      STATE.channelDetectionAttempts = 0;
      STATE.lastChannelDetectionAttempt = 0;
      STATE.watchTimeStart = Date.now();
      STATE.completionDetectionMethod = null;
      STATE.videoChangeDetected = true;
      STATE.badVideoCount = 0;
      STATE.lastBadVideoTimestamp = 0;
      STATE.userHasGivenFeedback = false; // Reset for new video
      STATE.aiActionBlocked = false; // Reset for new video
      STATE.stuckPercentageCount = 0; // Reset stuck counter
      STATE.lastPercentageValue = 0; // Reset last percentage
      
      // NEW: Reset user intent detection for new video
      STATE.userIntentToStay = false;
      STATE.userIntentDetected = false;
      STATE.scrollBackCount = 0;
      STATE.userOverrideActive = false;
      
      findVideoElement();
      setTimeout(() => extractMetadata(), 500);
      updateStatusOverlay();
      
      // Start completion checking interval
      STATE.completionCheckInterval = setInterval(() => {
        checkVideoCompletion();
      }, 1000);
      
      // NEW: Only set force scroll timer if user hasn't shown intent to stay
      if (CONFIG.forceScrollAfterSeconds > 0) {
        STATE.forceScrollTimer = setTimeout(() => {
          if (!STATE.videoCompleted && 
              !STATE.userIntentToStay && 
              !STATE.userOverrideActive && 
              STATE.currentVideoId === videoId) {
            console.log(`[ShortsAI] Force scrolling after ${CONFIG.forceScrollAfterSeconds} seconds`);
            STATE.lastAutoScrollTrigger = 'force_timer';
            scrollToNextVideo();
          } else if (STATE.userOverrideActive) {
            console.log("[ShortsAI] Force scroll cancelled - user override active");
          }
        }, CONFIG.forceScrollAfterSeconds * 1000);
      }
    }
  } catch (error) {
    console.error('[ShortsAI] Error in detectCurrentVideo:', error);
  }
}

// Enhanced completion detection
function checkVideoCompletion() {
  try {
    if (STATE.videoCompleted) return;
    
    if (!STATE.videoElement) {
      findVideoElement();
      return;
    }
    
    let shouldComplete = false;
    let completionReason = '';
    
    // Method 1: Video ended
    if (STATE.videoElement.ended) {
      shouldComplete = true;
      completionReason = 'ended';
    }
    
    // Method 2: Percentage threshold (lowered to 85%)
    else if (STATE.watchedPercent >= CONFIG.videoCompletionThreshold) {
      shouldComplete = true;
      completionReason = 'percentage';
    }
    
    // Method 3: Detect stuck percentage around 90%
    else if (STATE.watchedPercent >= 88) {
      if (Math.abs(STATE.watchedPercent - STATE.lastPercentageValue) < 1) {
        STATE.stuckPercentageCount++;
        console.log(`[ShortsAI] Percentage stuck at ${STATE.watchedPercent.toFixed(1)}% for ${STATE.stuckPercentageCount} seconds`);
        
        if (STATE.stuckPercentageCount >= 3) {
          shouldComplete = true;
          completionReason = 'stuck_percentage';
        }
      } else {
        STATE.stuckPercentageCount = 0;
      }
      STATE.lastPercentageValue = STATE.watchedPercent;
    }
    
    // Method 4: Near end and stalled
    else if (STATE.videoElement.duration > 0) {
      const timeRemaining = STATE.videoElement.duration - STATE.videoElement.currentTime;
      if (timeRemaining < 1 && STATE.watchedPercent > 80) {
        shouldComplete = true;
        completionReason = 'stalled_near_end';
      }
    }
    
    if (shouldComplete) {
      handleVideoCompletion(completionReason);
    }
  } catch (error) {
    console.error('[ShortsAI] Error in checkVideoCompletion:', error);
  }
}

// Enhanced completion handler that respects user intent
function handleVideoCompletion(method) {
  try {
    if (STATE.videoCompleted) return;
    
    console.log(`[ShortsAI] Video completed via ${method}`);
    STATE.videoCompleted = true;
    STATE.completionDetectionMethod = method;
    
    // Clear completion checking
    if (STATE.completionCheckInterval) {
      clearInterval(STATE.completionCheckInterval);
      STATE.completionCheckInterval = null;
    }
    
    sendEvent('completed', STATE.watchedPercent);
    
    // NEW: Only auto-scroll if user hasn't shown intent to stay
    if (STATE.autoScrollEnabled && 
        CONFIG.autoScrollOnCompletion && 
        !STATE.userIntentToStay && 
        !STATE.userOverrideActive) {
      console.log("[ShortsAI] GUARANTEED AUTO-SCROLL: Video completed, scrolling to next");
      STATE.lastAutoScrollTrigger = `completion_${method}`;
      STATE.lastAutoScrolledVideoId = STATE.currentVideoId;
      
      setTimeout(() => scrollToNextVideo(), 300);
      setTimeout(() => scrollToNextVideo(), 800);
      setTimeout(() => scrollToNextVideo(), 1500);
      setTimeout(() => scrollToNextVideo(), 2500);
    } else if (STATE.userOverrideActive) {
      console.log("[ShortsAI] Auto-scroll cancelled - user override active");
    }
  } catch (error) {
    console.error('[ShortsAI] Error in handleVideoCompletion:', error);
  }
}

function checkVideoState() {
  try {
    detectCurrentVideo();
    
    if (!STATE.videoElement || Date.now() - STATE.lastVideoElementCheck > 2000) {
      STATE.lastVideoElementCheck = Date.now();
      findVideoElement();
    }
    
    if (!STATE.metadataExtracted && STATE.currentVideoId) {
      const now = Date.now();
      if (now - STATE.lastChannelDetectionAttempt > 500 && STATE.channelDetectionAttempts < CONFIG.channelDetectionAttempts) {
        extractMetadata();
      }
    }
    
    if (STATE.metadataExtracted && STATE.score === 0 && STATE.currentVideoId) {
      getPrediction();
    }
    
    // NEW: Only apply auto-feedback if user hasn't taken control OR shown intent to stay
    if (STATE.autoFeedbackEnabled && 
        !STATE.pendingAutoFeedback && 
        !STATE.userHasGivenFeedback &&
        !STATE.aiActionBlocked &&
        !STATE.userIntentToStay &&
        !STATE.userOverrideActive &&
        STATE.score !== 0 && 
        STATE.watchedPercent >= CONFIG.feedbackWatchThreshold) {
      checkAutoFeedback();
    }
    
    // Check if auto-feedback should be confirmed (user didn't undo it)
    if (STATE.autoFeedbackType && 
        !STATE.autoFeedbackConfirmed && 
        STATE.autoFeedbackTimestamp > 0 &&
        Date.now() - STATE.autoFeedbackTimestamp > CONFIG.autoFeedbackConfirmationTime) {
      confirmAutoFeedback();
    }
  } catch (error) {
    console.error('[ShortsAI] Error in checkVideoState:', error);
  }
}

// Enhanced auto-feedback confirmation
function confirmAutoFeedback() {
  try {
    if (STATE.autoFeedbackConfirmed || !STATE.autoFeedbackType) return;
    
    console.log(`[ShortsAI] Auto-${STATE.autoFeedbackType} confirmed - user didn't undo it within ${CONFIG.autoFeedbackConfirmationTime/1000} seconds`);
    STATE.autoFeedbackConfirmed = true;
    
    // Send confirmation to AI with higher positive reinforcement
    if (STATE.autoFeedbackType === 'like') {
      sendEvent("auto_like_confirmed", 8);
    } else if (STATE.autoFeedbackType === 'dislike') {
      sendEvent("auto_dislike_confirmed", 8);
    }
  } catch (error) {
    console.error('[ShortsAI] Error in confirmAutoFeedback:', error);
  }
}

function findVideoElement() {
  try {
    const videoElements = document.querySelectorAll('video');
    let foundElement = null;
    
    for (const video of videoElements) {
      if (video.offsetWidth > 0 && video.offsetHeight > 0) {
        foundElement = video;
        break;
      }
    }
    
    if (foundElement && foundElement !== STATE.videoElement) {
      console.log('[ShortsAI] Found new video element:', foundElement);
      if (STATE.videoElement) {
        STATE.videoElement.removeEventListener('ended', handleVideoEnded);
        STATE.videoElement.removeEventListener('timeupdate', handleTimeUpdate);
        STATE.videoElement.removeEventListener('stalled', handleVideoStalled);
      }
      STATE.videoElement = foundElement;
      STATE.videoElement.addEventListener('ended', handleVideoEnded);
      STATE.videoElement.addEventListener('timeupdate', handleTimeUpdate);
      STATE.videoElement.addEventListener('stalled', handleVideoStalled);
    } else if (!foundElement && videoElements.length > 0 && !STATE.videoElement) {
      STATE.videoElement = videoElements[0];
      STATE.videoElement.addEventListener('ended', handleVideoEnded);
      STATE.videoElement.addEventListener('timeupdate', handleTimeUpdate);
      STATE.videoElement.addEventListener('stalled', handleVideoStalled);
      console.log('[ShortsAI] Using first video element as fallback:', STATE.videoElement);
    }
  } catch (error) {
    console.error('[ShortsAI] Error in findVideoElement:', error);
  }
}

function handleVideoEnded() {
  try {
    if (!STATE.videoCompleted) {
      handleVideoCompletion('ended_event');
    }
  } catch (error) {
    console.error('[ShortsAI] Error in handleVideoEnded:', error);
  }
}

function handleTimeUpdate() {
  try {
    if (!STATE.videoElement || STATE.videoCompleted) return;
    // Completion checking is now handled by checkVideoCompletion()
  } catch (error) {
    console.error('[ShortsAI] Error in handleTimeUpdate:', error);
  }
}

function updateWatchedPercent() {
  try {
    if (!STATE.videoElement || STATE.videoCompleted) return;

    const now = Date.now();
    if (now - STATE.lastWatchedPercentUpdate < CONFIG.watchedPercentUpdateInterval) return;
    STATE.lastWatchedPercentUpdate = now;

    if (STATE.videoElement.duration > 0 && !isNaN(STATE.videoElement.duration) && isFinite(STATE.videoElement.duration)) {
      const newPercent = (STATE.videoElement.currentTime / STATE.videoElement.duration) * 100;

      if (isNaN(newPercent) || !isFinite(newPercent)) {
        console.warn('[ShortsAI] Invalid watched percentage calculated:', newPercent);
        return;
      }

      if (newPercent < STATE.lastValidWatchedPercent - 20 && STATE.lastValidWatchedPercent > 10) {
        console.log(`[ShortsAI] Watched percent glitch detected (dropped from ${STATE.lastValidWatchedPercent.toFixed(1)}% to ${newPercent.toFixed(1)}%). Using max: ${STATE.maxWatchedPercent.toFixed(1)}%`);
        STATE.watchedPercent = STATE.maxWatchedPercent;
      } else if (newPercent < 5 && STATE.maxWatchedPercent > 50) {
        console.log(`[ShortsAI] Video restart detected. Using max watched: ${STATE.maxWatchedPercent.toFixed(1)}%`);
        STATE.watchedPercent = STATE.maxWatchedPercent;
      } else {
        STATE.watchedPercent = newPercent;
        STATE.lastValidWatchedPercent = newPercent;
      }

      STATE.maxWatchedPercent = Math.max(STATE.maxWatchedPercent, STATE.watchedPercent);

    } else {
      STATE.watchedPercent = 0;
      STATE.lastValidWatchedPercent = 0;
    }

    updateStatusOverlay();
  } catch (error) {
    console.error('[ShortsAI] Error in updateWatchedPercent:', error);
  }
}

function handleVideoStalled() {
  try {
    if (!STATE.videoCompleted) {
      if (STATE.videoElement && STATE.videoElement.duration > 0) {
        const timeRemaining = STATE.videoElement.duration - STATE.videoElement.currentTime;
        if (timeRemaining < 0.5 && STATE.watchedPercent > 80) {
          handleVideoCompletion('stalled_event');
        }
      }
    }
  } catch (error) {
    console.error('[ShortsAI] Error in handleVideoStalled:', error);
  }
}

function extractMetadata(force = false) {
  try {
    STATE.channelDetectionAttempts++;
    STATE.lastChannelDetectionAttempt = Date.now();
    if (STATE.metadataExtracted && !force) {
      return false;
    }

    let channelFound = false;
    let detectedChannelId = null;

    const channelElement = document.querySelector('a[href^="/channel/"], a[href^="/@"], ytd-channel-name a, #channel-name a, #owner-text a');
    if (channelElement) {
      const channelHref = channelElement.getAttribute('href');
      if (channelHref && channelHref.startsWith('/channel/')) {
        detectedChannelId = channelHref.split('/channel/')[1];
        channelFound = true;
      } else if (channelHref && channelHref.startsWith('/@')) {
        detectedChannelId = channelHref.substring(2);
        channelFound = true;
      }
      if (channelFound) console.log('[ShortsAI] Channel detected (Method 1):', detectedChannelId);
    }

    if (!channelFound && CONFIG.aggressiveChannelDetection) {
      const channelNameElements = document.querySelectorAll('.ytd-channel-name, #channel-name, #text-container.ytd-channel-name, #owner-text a, #owner a');
      for (const element of channelNameElements) {
        if (element.textContent && element.textContent.trim()) {
          detectedChannelId = element.textContent.trim();
          channelFound = true;
          console.log('[ShortsAI] Channel detected (Method 2):', detectedChannelId);
          break;
        }
      }
    }

    if (!channelFound && CONFIG.aggressiveChannelDetection) {
      const authorLinks = document.querySelectorAll('a.yt-simple-endpoint, #author-text a, #channel-info a, #owner a');
      for (const link of authorLinks) {
        try {
          if (link.href && (link.href.includes('/channel/') || link.href.includes('/@'))) {
            const href = link.href;
            if (href.includes('/channel/')) {
              detectedChannelId = href.split('/channel/')[1].split('?')[0];
              channelFound = true;
            } else if (href.includes('/@')) {
              detectedChannelId = href.split('/@')[1].split('?')[0];
              channelFound = true;
            }
            if (channelFound) {
              console.log('[ShortsAI] Channel detected (Method 3):', detectedChannelId);
              break;
            }
          }
        } catch (error) {
          console.warn('[ShortsAI] Error processing author link:', error);
        }
      }
    }

    if (channelFound) {
      STATE.currentChannelId = detectedChannelId;
    }

    const titleElement = document.querySelector('h1.ytd-watch-metadata, .title.ytd-video-primary-info-renderer, #title h1, .title');
    if (titleElement && titleElement.textContent) {
      STATE.currentTitle = titleElement.textContent.trim();
      console.log('[ShortsAI] Title detected:', STATE.currentTitle);
    }

    const descriptionElement = document.querySelector('#description, .description, .content.ytd-video-secondary-info-renderer');
    if (descriptionElement && descriptionElement.textContent) {
      STATE.currentDescription = descriptionElement.textContent.trim();
      console.log('[ShortsAI] Description detected (length):', STATE.currentDescription.length);
    }

    if (channelFound || STATE.currentTitle || STATE.currentDescription) {
      STATE.metadataExtracted = true;
      console.log('[ShortsAI] Metadata extraction completed');
      updateStatusOverlay();
      return true;
    }

    return false;
  } catch (error) {
    console.error('[ShortsAI] Error in extractMetadata:', error);
    return false;
  }
}

// --- UI Components ---

function createStatusOverlay() {
  try {
    if (STATE.statusElement) {
      STATE.statusElement.remove();
    }

    STATE.statusElement = document.createElement('div');
    STATE.statusElement.id = 'shorts-ai-status';
    STATE.statusElement.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 20px;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 10px;
      border-radius: 8px;
      font-family: Arial, sans-serif;
      font-size: 12px;
      z-index: 10000;
      max-width: 300px;
      border: 1px solid #333;
    `;

    const moodDropdown = document.createElement('select');
    moodDropdown.id = 'mood-selector';
    moodDropdown.style.cssText = `
      margin-bottom: 8px;
      padding: 4px;
      border-radius: 4px;
      border: 1px solid #555;
      background: #222;
      color: white;
      width: 100%;
    `;

    CONFIG.moods.forEach(mood => {
      const option = document.createElement('option');
      option.value = mood;
      option.textContent = mood;
      if (mood === STATE.currentMood) {
        option.selected = true;
      }
      moodDropdown.appendChild(option);
    });

    moodDropdown.addEventListener('change', (e) => {
      STATE.currentMood = e.target.value;
      console.log('[ShortsAI] Mood changed to:', STATE.currentMood);
      sendEvent('mood_change', 0, STATE.currentMood);
      updateStatusOverlay();
    });

    const statusText = document.createElement('div');
    statusText.id = 'status-text';

    STATE.statusElement.appendChild(moodDropdown);
    STATE.statusElement.appendChild(statusText);
    document.body.appendChild(STATE.statusElement);

    updateStatusOverlay();
    console.log('[ShortsAI] Status overlay created');
  } catch (error) {
    console.error('[ShortsAI] Error creating status overlay:', error);
  }
}

function updateStatusOverlay() {
  try {
    if (!STATE.statusElement) return;

    const statusText = STATE.statusElement.querySelector('#status-text');
    if (!statusText) return;

    let status = `Mood: ${STATE.currentMood}  
`;
    status += `Video: ${STATE.currentVideoId || 'None'}  
`;
    status += `Channel: ${STATE.currentChannelId || 'Detecting...'}  
`;
    status += `Score: ${STATE.score}  
`;
    status += `Watched: ${STATE.watchedPercent.toFixed(1)}%  
`;
    status += `Auto-scroll: ${STATE.autoScrollEnabled ? 'ON' : 'OFF'}  
`;
    status += `Auto-feedback: ${STATE.autoFeedbackEnabled ? 'ON' : 'OFF'}`;
    
    if (STATE.userHasGivenFeedback) {
      status += `  
<span style="color: #4CAF50;">User Control Active</span>`;
    }
    
    // NEW: Show user intent status
    if (STATE.userOverrideActive) {
      status += `  
<span style="color: #FF9800;">🎯 User Override Active</span>`;
    }

    if (CONFIG.showBufferInOverlay) {
      status += `  
Buffer: ${STATE.bufferSize}`;
    }

    statusText.innerHTML = status;
  } catch (error) {
    console.error('[ShortsAI] Error updating status overlay:', error);
  }
}

function createTrustButton() {
  try {
    if (STATE.trustButton) {
      STATE.trustButton.remove();
    }

    STATE.trustButton = document.createElement('div');
    STATE.trustButton.id = 'shorts-ai-trust-button';
    STATE.trustButton.style.cssText = `
      position: fixed;
      top: 50%;
      right: 20px;
      transform: translateY(-50%);
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 8px 12px;
      border-radius: 6px;
      font-family: Arial, sans-serif;
      font-size: 11px;
      z-index: 10000;
      cursor: pointer;
      border: 1px solid #333;
      display: none;
    `;

    STATE.trustButton.addEventListener('click', () => {
      if (STATE.currentChannelId) {
        if (STATE.channelIsTrusted) {
          sendEvent('untrust_channel', 0);
        } else if (STATE.channelIsBlocked) {
          sendEvent('unblock_channel', 0);
        } else {
          sendEvent('trust_channel', 0);
        }
      }
    });

    document.body.appendChild(STATE.trustButton);
    console.log('[ShortsAI] Trust button created');
  } catch (error) {
    console.error('[ShortsAI] Error creating trust button:', error);
  }
}

function updateTrustButton() {
  try {
    if (!STATE.trustButton || !STATE.currentChannelId) {
      if (STATE.trustButton) {
        STATE.trustButton.style.display = 'none';
      }
      return;
    }

    const now = Date.now();
    if (now - STATE.lastTrustButtonUpdate < CONFIG.trustButtonUpdateInterval) return;
    STATE.lastTrustButtonUpdate = now;

    if (now - STATE.lastChannelStatusCheck > 5000) {
      checkChannelStatus();
      STATE.lastChannelStatusCheck = now;
    }

    let buttonText = '';
    let buttonColor = '';

    if (STATE.channelIsTrusted) {
      buttonText = '✓ Trusted';
      buttonColor = '#4CAF50';
    } else if (STATE.channelIsBlocked) {
      buttonText = '✗ Blocked';
      buttonColor = '#f44336';
    } else {
      buttonText = '+ Trust';
      buttonColor = '#2196F3';
    }

    STATE.trustButton.textContent = buttonText;
    STATE.trustButton.style.background = buttonColor;
    STATE.trustButton.style.display = 'block';
  } catch (error) {
    console.error('[ShortsAI] Error updating trust button:', error);
  }
}

function setupKeyboardShortcuts() {
  try {
    document.addEventListener('keydown', (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
      }

      switch (event.key.toLowerCase()) {
        case 'a':
          if (event.ctrlKey) {
            event.preventDefault();
            STATE.autoScrollEnabled = !STATE.autoScrollEnabled;
            console.log('[ShortsAI] Auto-scroll toggled:', STATE.autoScrollEnabled);
            updateStatusOverlay();
          }
          break;
        case 'f':
          if (event.ctrlKey) {
            event.preventDefault();
            STATE.autoFeedbackEnabled = !STATE.autoFeedbackEnabled;
            console.log('[ShortsAI] Auto-feedback toggled:', STATE.autoFeedbackEnabled);
            updateStatusOverlay();
          }
          break;
        case 's':
          if (event.ctrlKey) {
            event.preventDefault();
            scrollToNextVideo();
          }
          break;
        case 'l':
          if (event.ctrlKey) {
            event.preventDefault();
            likeCurrentVideo();
          }
          break;
        case 'd':
          if (event.ctrlKey) {
            event.preventDefault();
            dislikeCurrentVideo();
          }
          break;
        case 't':
          if (event.ctrlKey && STATE.currentChannelId) {
            event.preventDefault();
            if (STATE.channelIsTrusted) {
              sendEvent('untrust_channel', 0);
            } else {
              sendEvent('trust_channel', 0);
            }
          }
          break;
        case 'b':
          if (event.ctrlKey && STATE.currentChannelId) {
            event.preventDefault();
            if (STATE.channelIsBlocked) {
              sendEvent('unblock_channel', 0);
            } else {
              sendEvent('block_channel', 0);
            }
          }
          break;
      }
    });

    console.log('[ShortsAI] Keyboard shortcuts set up');
  } catch (error) {
    console.error('[ShortsAI] Error setting up keyboard shortcuts:', error);
  }
}

// --- API Communication ---

function sendEvent(eventType, watchedPercent = STATE.watchedPercent, mood = STATE.currentMood) {
  try {
    if (!STATE.currentVideoId) {
      console.warn('[ShortsAI] Cannot send event: no current video ID');
      return;
    }

    const eventData = {
      video_id: STATE.currentVideoId,
      channel_id: STATE.currentChannelId || 'unknown',
      title: STATE.currentTitle || '',
      description: STATE.currentDescription || '',
      captions: STATE.currentCaptions || '',
      event_type: eventType,
      watched_percent: watchedPercent,
      mood: mood
    };

    console.log('[ShortsAI] Sending event:', eventType, eventData);

    fetch(`${CONFIG.apiBaseUrl}/event`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(eventData)
    })
    .then(response => response.json())
    .then(data => {
      console.log('[ShortsAI] Event response:', data);
      if (data.corrections_made !== undefined) {
        STATE.correctionsMade = data.corrections_made;
      }
    })
    .catch(error => {
      console.error('[ShortsAI] Error sending event:', error);
    });
  } catch (error) {
    console.error('[ShortsAI] Error in sendEvent:', error);
  }
}

function getPrediction() {
  try {
    if (!STATE.currentVideoId || !STATE.metadataExtracted) {
      return;
    }

    const metadata = {
      video_id: STATE.currentVideoId,
      channel_id: STATE.currentChannelId || 'unknown',
      title: STATE.currentTitle || '',
      description: STATE.currentDescription || '',
      captions: STATE.currentCaptions || ''
    };

    console.log('[ShortsAI] Getting prediction for:', metadata);

    fetch(`${CONFIG.apiBaseUrl}/next`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(metadata)
    })
    .then(response => response.json())
    .then(data => {
      console.log('[ShortsAI] Prediction response:', data);
      STATE.score = data.score || 0;
      
      if (data.mood_suggestion && data.mood_suggestion !== STATE.currentMood) {
        console.log('[ShortsAI] AI suggests mood change to:', data.mood_suggestion);
      }
      
      updateStatusOverlay();
      
      // NEW: Only auto-skip if user hasn't shown intent to stay
      if (STATE.score <= CONFIG.autoSkipThreshold && 
          STATE.autoScrollEnabled && 
          !STATE.userIntentToStay && 
          !STATE.userOverrideActive) {
        console.log('[ShortsAI] Auto-skipping video with low score:', STATE.score);
        STATE.lastAutoScrollTrigger = 'low_score';
        STATE.lastAutoScrolledVideoId = STATE.currentVideoId;
        
        const delay = STATE.badVideoCount > 0 && 
                     Date.now() - STATE.lastBadVideoTimestamp < 10000 ? 
                     CONFIG.badVideoScrollDelay : 0;
        
        STATE.badVideoCount++;
        STATE.lastBadVideoTimestamp = Date.now();
        
        setTimeout(() => {
          scrollToNextVideo();
        }, delay);
      } else if (STATE.userOverrideActive) {
        console.log('[ShortsAI] Auto-skip cancelled - user override active');
      }
    })
    .catch(error => {
      console.error('[ShortsAI] Error getting prediction:', error);
    });
  } catch (error) {
    console.error('[ShortsAI] Error in getPrediction:', error);
  }
}

function checkChannelStatus() {
  try {
    if (!STATE.currentChannelId) return;

    fetch(`${CONFIG.apiBaseUrl}/channel_status?channel_id=${encodeURIComponent(STATE.currentChannelId)}`)
    .then(response => response.json())
    .then(data => {
      STATE.channelIsTrusted = data.trusted || false;
      STATE.channelIsBlocked = data.blocked || false;
      console.log('[ShortsAI] Channel status:', {
        trusted: STATE.channelIsTrusted,
        blocked: STATE.channelIsBlocked
      });
    })
    .catch(error => {
      console.error('[ShortsAI] Error checking channel status:', error);
    });
  } catch (error) {
    console.error('[ShortsAI] Error in checkChannelStatus:', error);
  }
}

function fetchBufferSize() {
  try {
    fetch(`${CONFIG.apiBaseUrl}/buffer_size`)
    .then(response => response.json())
    .then(data => {
      STATE.bufferSize = data.buffer_size || CONFIG.bufferSize;
      console.log('[ShortsAI] Buffer size:', STATE.bufferSize);
      updateStatusOverlay();
    })
    .catch(error => {
      console.error('[ShortsAI] Error fetching buffer size:', error);
    });
  } catch (error) {
    console.error('[ShortsAI] Error in fetchBufferSize:', error);
  }
}

// --- Video Actions ---

// NEW: Enhanced scroll function that respects user intent
function scrollToNextVideo() {
  try {
    // Don't scroll if user has shown intent to stay
    if (STATE.userIntentToStay || STATE.userOverrideActive) {
      console.log('[ShortsAI] Scroll blocked - user wants to stay on this video');
      return;
    }
    
    const now = Date.now();
    if (now - STATE.lastScrollTime < CONFIG.minScrollInterval && !CONFIG.zeroDelayScroll) {
      console.log('[ShortsAI] Scroll rate limited');
      return;
    }

    STATE.lastScrollTime = now;
    STATE.scrollAttempts++;

    console.log('[ShortsAI] SCROLLING TO NEXT VIDEO (attempt', STATE.scrollAttempts, ')');

    const methods = [
      () => {
        document.dispatchEvent(new KeyboardEvent('keydown', { 
          key: 'ArrowDown', 
          keyCode: 40,
          which: 40,
          bubbles: true,
          cancelable: true
        }));
        console.log('[ShortsAI] Scroll method: ArrowDown key');
      },
      () => {
        window.scrollBy({
          top: window.innerHeight,
          behavior: 'smooth'
        });
        console.log('[ShortsAI] Scroll method: smooth scroll');
      },
      () => {
        window.scrollBy(0, window.innerHeight);
        console.log('[ShortsAI] Scroll method: immediate scroll');
      },
      () => {
        const nextButton = document.querySelector('button[aria-label*="Next"], button[aria-label*="next"]');
        if (nextButton) {
          nextButton.click();
          console.log('[ShortsAI] Scroll method: next button click');
        } else {
          window.scrollBy(0, window.innerHeight * 1.5);
          console.log('[ShortsAI] Scroll method: fallback large scroll');
        }
      }
    ];

    const methodIndex = STATE.scrollAttempts % methods.length;
    methods[methodIndex]();

    sendEvent('manual_skip', STATE.watchedPercent);
  } catch (error) {
    console.error('[ShortsAI] Error in scrollToNextVideo:', error);
  }
}

function likeCurrentVideo() {
  try {
    const likeButton = document.querySelector('yt-icon-button[aria-label*="like"], button[aria-label*="like"]');
    if (likeButton) {
      likeButton.click();
      console.log('[ShortsAI] Liked video');
      sendEvent('like', STATE.watchedPercent);
    }
  } catch (error) {
    console.error('[ShortsAI] Error liking video:', error);
  }
}

function dislikeCurrentVideo() {
  try {
    const dislikeButton = document.querySelector('yt-icon-button[aria-label*="dislike"], button[aria-label*="dislike"]');
    if (dislikeButton) {
      dislikeButton.click();
      console.log('[ShortsAI] Disliked video');
      sendEvent('dislike', STATE.watchedPercent);
    }
  } catch (error) {
    console.error('[ShortsAI] Error disliking video:', error);
  }
}

// Enhanced auto-feedback that respects user intent
function checkAutoFeedback() {
  try {
    if (STATE.pendingAutoFeedback || 
        STATE.userHasGivenFeedback || 
        STATE.aiActionBlocked ||
        STATE.userIntentToStay ||
        STATE.userOverrideActive) {
      return;
    }

    // Double-check that user hasn't already given feedback
    const likeButton = document.querySelector('yt-icon-button[aria-label*="like"], button[aria-label*="like"]');
    const dislikeButton = document.querySelector('yt-icon-button[aria-label*="dislike"], button[aria-label*="dislike"]');
    
    const isAlreadyLiked = likeButton && (
      likeButton.getAttribute('aria-pressed') === 'true' || 
      likeButton.classList.contains('style-default-active')
    );
    const isAlreadyDisliked = dislikeButton && (
      dislikeButton.getAttribute('aria-pressed') === 'true' || 
      dislikeButton.classList.contains('style-default-active')
    );
    
    if (isAlreadyLiked || isAlreadyDisliked) {
      console.log('[ShortsAI] Video already has manual feedback, blocking AI');
      STATE.userHasGivenFeedback = true;
      STATE.aiActionBlocked = true;
      return;
    }

    if (STATE.score >= CONFIG.likeThreshold) {
      STATE.pendingAutoFeedback = true;
      STATE.autoFeedbackType = 'like';
      STATE.autoFeedbackTimestamp = Date.now();
      STATE.autoFeedbackConfirmed = false;
      likeCurrentVideo();
      console.log(`[ShortsAI] AUTO-LIKED video with score: ${STATE.score} (threshold: ${CONFIG.likeThreshold})`);
      
      STATE.autoFeedbackConfirmationTimer = setTimeout(() => {
        confirmAutoFeedback();
      }, CONFIG.autoFeedbackConfirmationTime);
      
    } else if (STATE.score <= CONFIG.dislikeThreshold) {
      STATE.pendingAutoFeedback = true;
      STATE.autoFeedbackType = 'dislike';
      STATE.autoFeedbackTimestamp = Date.now();
      STATE.autoFeedbackConfirmed = false;
      dislikeCurrentVideo();
      console.log(`[ShortsAI] AUTO-DISLIKED video with score: ${STATE.score} (threshold: ${CONFIG.dislikeThreshold})`);
      
      STATE.autoFeedbackConfirmationTimer = setTimeout(() => {
        confirmAutoFeedback();
      }, CONFIG.autoFeedbackConfirmationTime);
    }
  } catch (error) {
    console.error('[ShortsAI] Error in checkAutoFeedback:', error);
  }
}

function handleManualScroll() {
  try {
    if (STATE.currentVideoId && !STATE.videoCompleted) {
      console.log('[ShortsAI] Manual scroll detected');
      sendEvent('manual_skip', STATE.watchedPercent);
    }
  } catch (error) {
    console.error('[ShortsAI] Error in handleManualScroll:', error);
  }
}

console.log("[ShortsAI] Content script loaded successfully (User Intent Detection Version)");
