// background.js - Chrome Extension Background Script

console.log("[ShortsAI] Background script loaded");

// Handle extension installation
chrome.runtime.onInstalled.addListener(() => {
  console.log("[ShortsAI] Extension installed");
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("[ShortsAI] Message received:", request);
  
  if (request.action === "log") {
    console.log("[ShortsAI Content]", request.message);
  }
  
  sendResponse({status: "received"});
});

// Optional: Handle tab updates to inject content script
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && tab.url.includes('youtube.com/shorts')) {
    console.log("[ShortsAI] YouTube Shorts page detected:", tab.url);
  }
});
