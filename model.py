import json
import os
import time
import random
import glob
import re
from collections import deque

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

from features import FeatureBuilder


if nn is not None:
    class _SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )

        def forward(self, x):
            return self.net(x)
else:
    _SimpleClassifier = None

class ShortsAIModel:
    def __init__(self, data_file='shorts_ai_data.json'):
        self.data_file = data_file
        self.feature_builder = FeatureBuilder('vocab.json')
        self.base_model = None
        self.user_model = None
        self.user_model_path = 'UserModel.pt'
        self.inference_available = torch is not None and self.feature_builder.total_dim > 0
        self.user_preferences = {
            'video_scores': {},
            'channel_scores': {},
            'trusted_channels': set(),
            'blocked_channels': set(),
            'mood_preferences': {
                'Neutral': {'video_scores': {}, 'channel_scores': {}},
                'Happy': {'video_scores': {}, 'channel_scores': {}},
                'Relaxed': {'video_scores': {}, 'channel_scores': {}},
                'Focused': {'video_scores': {}, 'channel_scores': {}},
                'Energetic': {'video_scores': {}, 'channel_scores': {}},
                'Curious': {'video_scores': {}, 'channel_scores': {}},
                'Creative': {'video_scores': {}, 'channel_scores': {}},
                'Mad': {'video_scores': {}, 'channel_scores': {}}
            },
            'current_mood': 'Neutral',
            'mood_last_changed': time.time(),
            'mood_history': deque(maxlen=10),
            'previously_watched_videos': {} # Track previously watched videos
        }
        self.load_data()
        self.buffer = deque(maxlen=20)
        self.last_mood_check = time.time()
        self._load_models()
        
        # Enhanced learning rate for faster adaptation
        self.learning_rate = 0.3  # Increased from 0.25 to 0.3 for even faster learning

    def _find_highest_version_model(self):
        best_path = None
        best_ver = (-1, -1)
        for path in glob.glob('Model*.pt'):
            match = re.search(r'Model(\d+)\.(\d+)\.pt$', os.path.basename(path))
            if not match:
                continue
            ver = (int(match.group(1)), int(match.group(2)))
            if ver > best_ver:
                best_ver = ver
                best_path = path
        return best_path

    def _safe_load_torch_model(self, path):
        if not self.inference_available or not path or not os.path.exists(path):
            return None
        try:
            model = _SimpleClassifier(self.feature_builder.total_dim)
            loaded = torch.load(path, map_location='cpu')
            state_dict = loaded.get('state_dict', loaded) if isinstance(loaded, dict) else loaded
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            print(f"[Model] Could not load model from {path}: {e}")
            return None

    def _load_models(self):
        if not self.inference_available:
            print('[Model] Torch/vocab unavailable; using rule-based fallback only')
            return

        base_path = self._find_highest_version_model()
        if base_path:
            self.base_model = self._safe_load_torch_model(base_path)
            if self.base_model:
                print(f'[Model] Loaded base model: {base_path}')

        self.user_model = self._safe_load_torch_model(self.user_model_path)
        if self.user_model:
            print(f'[Model] Loaded user model: {self.user_model_path}')

    def _rule_based_probabilities(self, score):
        score = max(-100.0, min(100.0, float(score)))
        p_like = max(0.0, score) / 100.0
        p_dislike = max(0.0, -score) / 100.0
        p_neutral = max(0.0, 1.0 - (p_like + p_dislike))
        total = p_like + p_dislike + p_neutral or 1.0
        return {
            'p_like': p_like / total,
            'p_dislike': p_dislike / total,
            'p_neutral': p_neutral / total,
        }

    def _predict_probabilities(self, title='', description='', subtitles='', category='', duration_seconds=None):
        if not self.inference_available:
            return None
        if not self.base_model and not self.user_model:
            return None
        try:
            vector = self.feature_builder.build_vector(
                title=title,
                description=description,
                subtitles=subtitles,
                category=category,
                duration_seconds=duration_seconds,
                timestamp=time.time(),
            )
            x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            logits = torch.zeros((1, 3), dtype=torch.float32)
            if self.base_model:
                self.base_model.eval()
                logits = logits + self.base_model(x)
            if self.user_model:
                self.user_model.eval()
                logits = logits + 0.6 * self.user_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
            return {'p_like': probs[0], 'p_dislike': probs[1], 'p_neutral': probs[2]}
        except Exception as e:
            print(f'[Model] Inference failed, using fallback: {e}')
            return None

    def _probabilities_to_decision(self, p_like, p_dislike, p_neutral):
        if p_like >= 0.62 and p_like > p_dislike:
            return 'like'
        if p_dislike >= 0.62 and p_dislike > p_like:
            return 'dislike'
        if p_dislike >= 0.45 and p_like <= 0.30:
            return 'skip'
        return 'neutral'

    def _tiny_incremental_finetune(self, event_type, title='', description='', subtitles='', category='', duration_seconds=None):
        if torch is None:
            return

        label_map = {'user_like': 0, 'like': 0, 'user_dislike': 1, 'dislike': 1}
        if event_type not in label_map:
            return

        if not self.user_model:
            if self.base_model:
                self.user_model = _SimpleClassifier(self.feature_builder.total_dim)
                self.user_model.load_state_dict(self.base_model.state_dict(), strict=False)
            elif self.inference_available:
                self.user_model = _SimpleClassifier(self.feature_builder.total_dim)
            else:
                return

        try:
            vector = self.feature_builder.build_vector(
                title=title,
                description=description,
                subtitles=subtitles,
                category=category,
                duration_seconds=duration_seconds,
                timestamp=time.time(),
            )
            x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor([label_map[event_type]], dtype=torch.long)

            self.user_model.train()
            optimizer = torch.optim.Adam(self.user_model.parameters(), lr=5e-5)
            criterion = nn.CrossEntropyLoss()

            for _ in range(random.randint(1, 3)):
                optimizer.zero_grad()
                logits = self.user_model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            self.user_model.eval()
            torch.save({'state_dict': self.user_model.state_dict()}, self.user_model_path)
            print('[Model] Saved incremental user model update to UserModel.pt')
        except Exception as e:
            print(f'[Model] Incremental fine-tuning skipped due to error: {e}')

    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                # Load basic preferences
                self.user_preferences['video_scores'] = data.get('video_scores', {})
                self.user_preferences['channel_scores'] = data.get('channel_scores', {})
                self.user_preferences['trusted_channels'] = set(data.get('trusted_channels', []))
                self.user_preferences['blocked_channels'] = set(data.get('blocked_channels', []))
                
                # Load mood preferences
                mood_prefs = data.get('mood_preferences', {})
                for mood in self.user_preferences['mood_preferences']:
                    if mood in mood_prefs:
                        self.user_preferences['mood_preferences'][mood] = mood_prefs[mood]
                
                # Load mood state
                self.user_preferences['current_mood'] = data.get('current_mood', 'Neutral')
                self.user_preferences['mood_last_changed'] = data.get('mood_last_changed', time.time())
                
                # Load mood history
                mood_history = data.get('mood_history', [])
                self.user_preferences['mood_history'] = deque(mood_history, maxlen=10)
                
                # Load previously watched videos
                self.user_preferences['previously_watched_videos'] = data.get('previously_watched_videos', {})
                
                print(f"[Model] Loaded data from {self.data_file}")
            except Exception as e:
                print(f"[Model] Error loading data: {e}")
                self.save_data()
        else:
            print(f"[Model] No existing data file, starting fresh")
            self.save_data()

    def save_data(self):
        try:
            data = {
                'video_scores': self.user_preferences['video_scores'],
                'channel_scores': self.user_preferences['channel_scores'],
                'trusted_channels': list(self.user_preferences['trusted_channels']),
                'blocked_channels': list(self.user_preferences['blocked_channels']),
                'mood_preferences': self.user_preferences['mood_preferences'],
                'current_mood': self.user_preferences['current_mood'],
                'mood_last_changed': self.user_preferences['mood_last_changed'],
                'mood_history': list(self.user_preferences['mood_history']),
                'previously_watched_videos': self.user_preferences['previously_watched_videos']
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[Model] Saved data to {self.data_file}")
        except Exception as e:
            print(f"[Model] Error saving data: {e}")

    def get_current_mood(self):
        # Check for mood decay (Mad mood auto-decay)
        current_time = time.time()
        time_since_change = current_time - self.user_preferences['mood_last_changed']
        
        if self.user_preferences['current_mood'] == 'Mad':
            if time_since_change > 600:  # 10 minutes
                self.set_mood('Happy')
                print("[Model] Mad mood auto-decayed to Happy after 10 minutes")
            elif time_since_change > 480:  # 8 minutes
                self.set_mood('Relaxed')
                print("[Model] Mad mood auto-decayed to Relaxed after 8 minutes")
            elif time_since_change > 180:  # 3 minutes
                self.set_mood('Neutral')
                print("[Model] Mad mood auto-decayed to Neutral after 3 minutes")
        
        return self.user_preferences['current_mood']

    def set_mood(self, mood):
        if mood in self.user_preferences['mood_preferences']:
            old_mood = self.user_preferences['current_mood']
            self.user_preferences['current_mood'] = mood
            self.user_preferences['mood_last_changed'] = time.time()
            self.user_preferences['mood_history'].append(mood)
            print(f"[Model] Mood changed from {old_mood} to {mood}")
            self.save_data()

    def process_event(self, video_id, channel_id, event_type, watched_percent, mood='Neutral', title='', description='', captions='', category='', duration_seconds=None, algorithm_action=None, user_action=None):
        print(f"[Model] Processing event: {event_type} for video {video_id} (mood: {mood})")
        
        # Update current mood
        if mood != self.user_preferences['current_mood']:
            self.set_mood(mood)
        
        current_mood = self.get_current_mood()
        
        # Track previously watched videos
        if event_type in ['completed', 'user_like', 'user_dislike'] and watched_percent > 10:
            self.user_preferences['previously_watched_videos'][video_id] = {
                'watch_time': watched_percent,
                'timestamp': time.time(),
                'mood': current_mood
            }
        
        # Add to buffer for potential corrections
        self.buffer.append({
            'video_id': video_id,
            'channel_id': channel_id,
            'event_type': event_type,
            'watched_percent': watched_percent,
            'timestamp': time.time(),
            'mood': current_mood
        })
        
        corrections_made = 0
        
        # ENHANCED: Massive penalties for AI going against user intent
        if event_type == 'ai_scroll_against_user_intent':
            score_change = -100 * self.learning_rate  # MASSIVE penalty
            self.update_scores(video_id, channel_id, score_change, current_mood)
            print(f"[Model] 💥 MASSIVE PENALTY: AI scrolled against user intent! Score change: {score_change}")
        elif event_type == 'ai_scroll_blocked_by_user_intent':
            score_change = -100 * self.learning_rate  # MASSIVE penalty
            self.update_scores(video_id, channel_id, score_change, current_mood)
            print(f"[Model] 💥 MASSIVE PENALTY: AI scroll blocked by user intent! Score change: {score_change}")
        elif event_type == 'ai_scroll_blocked_user_recent_activity':
            score_change = -50 * self.learning_rate  # Heavy penalty
            self.update_scores(video_id, channel_id, score_change, current_mood)
            print(f"[Model] 🔥 HEAVY PENALTY: AI tried to scroll while user was active! Score change: {score_change}")
        
        # Enhanced user feedback with massive rewards
        elif event_type == 'user_like':
            score_change = 25 * self.learning_rate  # MASSIVE positive signal
            self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'user_dislike':
            score_change = -25 * self.learning_rate  # MASSIVE negative signal
            self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'user_intent_to_stay':
            score_change = 30 * self.learning_rate  # MASSIVE positive for user intent
            self.update_scores(video_id, channel_id, score_change, current_mood)
            print(f"[Model] 🎯 USER INTENT TO STAY: Massive positive reinforcement! Score change: {score_change}")
        
        # Enhanced penalties for wrong auto-actions
        elif event_type == 'undo_auto_like':
            score_change = -25 * self.learning_rate  # MASSIVE penalty for wrong auto-like
            self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'undo_auto_dislike':
            score_change = 25 * self.learning_rate  # MASSIVE positive for wrong auto-dislike
            self.update_scores(video_id, channel_id, score_change, current_mood)
        
        # Enhanced auto-feedback confirmations
        elif event_type == 'auto_like_confirmed':
            score_change = 15 * self.learning_rate  # Increased reward for correct auto-like
            self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'auto_dislike_confirmed':
            score_change = -15 * self.learning_rate  # Increased reward for correct auto-dislike
            self.update_scores(video_id, channel_id, score_change, current_mood)
        
        # Regular feedback events
        elif event_type == 'like':
            score_change = 18 * self.learning_rate  # Increased from 15
            self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'dislike':
            score_change = -18 * self.learning_rate  # Increased from -15
            self.update_scores(video_id, channel_id, score_change, current_mood)
        
        # Different handling for previously watched videos
        elif event_type == 'manual_skip_previously_watched':
            if watched_percent < 20:
                score_change = -3 * self.learning_rate  # Mild penalty for early skip of previously watched
                self.update_scores(video_id, channel_id, score_change, current_mood)
            # No penalty for skipping previously watched videos after some time
        elif event_type == 'user_scroll_away_previously_watched':
            # Neutral - this is expected behavior for previously watched videos
            pass
        elif event_type == 'user_early_scroll_away':
            score_change = -5 * self.learning_rate  # Mild penalty for early scroll away from new videos
            self.update_scores(video_id, channel_id, score_change, current_mood)
        
        # Regular manual skip
        elif event_type == 'manual_skip':
            if watched_percent < 30:
                score_change = -10 * self.learning_rate  # Increased penalty for early manual skip
                self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'completed':
            if watched_percent >= 80:
                score_change = 12 * self.learning_rate  # Increased reward for completion
                self.update_scores(video_id, channel_id, score_change, current_mood)
        
        # AI scroll events
        elif event_type == 'ai_scroll':
            # Mild negative if AI scrolled away from a video user might have wanted to watch
            score_change = -2 * self.learning_rate
            self.update_scores(video_id, channel_id, score_change, current_mood)
        elif event_type == 'ai_scroll_previously_watched':
            # Neutral - AI scrolling away from previously watched videos is fine
            pass
        
        # Channel management
        elif event_type == 'trust_channel':
            self.user_preferences['trusted_channels'].add(channel_id)
            self.user_preferences['blocked_channels'].discard(channel_id)
        elif event_type == 'untrust_channel':
            self.user_preferences['trusted_channels'].discard(channel_id)
        elif event_type == 'block_channel':
            self.user_preferences['blocked_channels'].add(channel_id)
            self.user_preferences['trusted_channels'].discard(channel_id)
        elif event_type == 'unblock_channel':
            self.user_preferences['blocked_channels'].discard(channel_id)
        
        if algorithm_action or user_action:
            print(f"[Model] Action log - algorithm_action={algorithm_action}, user_action={user_action}")

        self._tiny_incremental_finetune(
            event_type=event_type,
            title=title,
            description=description,
            subtitles=captions,
            category=category,
            duration_seconds=duration_seconds,
        )

        self.save_data()
        return {'corrections_made': corrections_made}

    def update_scores(self, video_id, channel_id, score_change, mood):
        # Update general scores
        if video_id not in self.user_preferences['video_scores']:
            self.user_preferences['video_scores'][video_id] = 0
        if channel_id not in self.user_preferences['channel_scores']:
            self.user_preferences['channel_scores'][channel_id] = 0
        
        self.user_preferences['video_scores'][video_id] += score_change
        self.user_preferences['channel_scores'][channel_id] += score_change * 0.5  # Channel gets half the impact
        
        # Update mood-specific scores
        mood_prefs = self.user_preferences['mood_preferences'][mood]
        if 'video_scores' not in mood_prefs:
            mood_prefs['video_scores'] = {}
        if 'channel_scores' not in mood_prefs:
            mood_prefs['channel_scores'] = {}
        
        if video_id not in mood_prefs['video_scores']:
            mood_prefs['video_scores'][video_id] = 0
        if channel_id not in mood_prefs['channel_scores']:
            mood_prefs['channel_scores'][channel_id] = 0
        
        mood_prefs['video_scores'][video_id] += score_change
        mood_prefs['channel_scores'][channel_id] += score_change * 0.5
        
        print(f"[Model] Updated scores for {video_id} (mood: {mood}): video={self.user_preferences['video_scores'][video_id]:.1f}, channel={self.user_preferences['channel_scores'][channel_id]:.1f}, change={score_change:.1f}")

    def predict_score(self, video_id, channel_id, title='', description='', captions='', category='', duration_seconds=None):
        current_mood = self.get_current_mood()
        
        # Check if this video was previously watched
        is_previously_watched = video_id in self.user_preferences['previously_watched_videos']
        
        # Base scores
        video_score = self.user_preferences['video_scores'].get(video_id, 0)
        channel_score = self.user_preferences['channel_scores'].get(channel_id, 0)
        
        # Mood-specific scores (higher weight)
        mood_prefs = self.user_preferences['mood_preferences'][current_mood]
        mood_video_score = mood_prefs.get('video_scores', {}).get(video_id, 0)
        mood_channel_score = mood_prefs.get('channel_scores', {}).get(channel_id, 0)
        
        # Combine scores with mood preference having higher weight
        combined_score = (video_score * 0.3 + mood_video_score * 0.7 + 
                         channel_score * 0.3 + mood_channel_score * 0.7)
        
        # Previously watched video handling
        if is_previously_watched:
            prev_data = self.user_preferences['previously_watched_videos'][video_id]
            time_since_watched = time.time() - prev_data['timestamp']
            
            # If watched recently (within 24 hours), slight negative bias
            if time_since_watched < 86400:  # 24 hours
                combined_score -= 5
                print(f"[Model] Previously watched video (recent): slight negative bias applied")
            # If watched long ago (over a week), neutral to slight positive
            elif time_since_watched > 604800:  # 1 week
                combined_score += 2
                print(f"[Model] Previously watched video (old): slight positive bias applied")
        
        # Channel status modifiers
        if channel_id in self.user_preferences['trusted_channels']:
            combined_score += 20  # Increased from 15
        elif channel_id in self.user_preferences['blocked_channels']:
            combined_score -= 30  # Increased from -25
        
        # Enhanced text analysis
        text_score = 0
        full_text = f"{title} {description}".lower()
        
        # Mood-based text analysis
        if current_mood == 'Happy':
            if any(word in full_text for word in ['funny', 'comedy', 'laugh', 'joke', 'humor', 'fun']):
                text_score += 10  # Increased from 8
        elif current_mood == 'Relaxed':
            if any(word in full_text for word in ['calm', 'peaceful', 'relax', 'chill', 'asmr', 'nature']):
                text_score += 10  # Increased from 8
        elif current_mood == 'Focused':
            if any(word in full_text for word in ['tutorial', 'learn', 'education', 'how to', 'guide', 'tips']):
                text_score += 10  # Increased from 8
        elif current_mood == 'Energetic':
            if any(word in full_text for word in ['workout', 'fitness', 'energy', 'motivation', 'sports', 'dance']):
                text_score += 10  # Increased from 8
        elif current_mood == 'Curious':
            if any(word in full_text for word in ['science', 'discovery', 'mystery', 'explore', 'facts', 'amazing']):
                text_score += 10  # Increased from 8
        elif current_mood == 'Creative':
            if any(word in full_text for word in ['art', 'creative', 'diy', 'craft', 'design', 'make']):
                text_score += 10  # Increased from 8
        elif current_mood == 'Mad':
            # For mad mood, initially allow some venting content but guide toward positive
            if any(word in full_text for word in ['rant', 'angry', 'frustrated']):
                text_score += 3  # Slight boost for venting content
            elif any(word in full_text for word in ['calm', 'peaceful', 'positive', 'happy']):
                text_score += 15  # Strong boost for mood-improving content
        
        # General positive/negative indicators
        if any(word in full_text for word in ['amazing', 'awesome', 'great', 'best', 'love']):
            text_score += 8  # Increased from 6
        if any(word in full_text for word in ['boring', 'stupid', 'hate', 'worst', 'terrible']):
            text_score -= 10  # Increased from -8
        if 'fail' in title.lower() or 'fail' in description.lower():
            text_score -= 10  # Increased from -8

        combined_score += text_score

        # Ensure score is within bounds
        final_score = max(-100, min(100, combined_score))

        status = "new" if not is_previously_watched else "previously_watched"
        print(f"[Model] Predicted score for {video_id} ({status}, mood: {current_mood}): {final_score:.2f} (Video: {video_score}, Channel: {channel_score}, Mood Video: {mood_video_score}, Mood Channel: {mood_channel_score}, Text: {text_score})")
        
        mood_suggestion = self.suggest_mood_change()

        probs = self._predict_probabilities(
            title=title,
            description=description,
            subtitles=captions,
            category=category,
            duration_seconds=duration_seconds,
        )
        if probs is None:
            probs = self._rule_based_probabilities(final_score)

        decision = self._probabilities_to_decision(
            probs['p_like'], probs['p_dislike'], probs['p_neutral']
        )

        return {
            'score': round(final_score),
            'p_like': round(probs['p_like'], 4),
            'p_dislike': round(probs['p_dislike'], 4),
            'p_neutral': round(probs['p_neutral'], 4),
            'decision': decision,
            'mood_suggestion': mood_suggestion
        }

    def get_channel_status(self, channel_id):
        trusted = channel_id in self.user_preferences['trusted_channels']
        blocked = channel_id in self.user_preferences['blocked_channels']
        print(f"[Model] Channel status for {channel_id}: Trusted={trusted}, Blocked={blocked}")
        return {'trusted': trusted, 'blocked': blocked}

    def get_buffer_size(self):
        return len(self.buffer)

    def suggest_mood_change(self):
        # Enhanced mood suggestion logic
        current_time = time.time()
        
        # Check if it's time for a mood suggestion (every 5 minutes)
        if current_time - self.last_mood_check > 300:  # 5 minutes
            self.last_mood_check = current_time
            
            # Analyze recent interactions for mood suggestions
            recent_events = [event for event in self.buffer if current_time - event['timestamp'] < 300]
            
            if len(recent_events) >= 3:
                positive_events = sum(1 for event in recent_events if event['event_type'] in ['like', 'user_like', 'completed', 'user_intent_to_stay'])
                negative_events = sum(1 for event in recent_events if event['event_type'] in ['dislike', 'user_dislike', 'manual_skip', 'ai_scroll_against_user_intent'])
                
                if negative_events > positive_events * 2:
                    # Suggest a more positive mood
                    return random.choice(['Happy', 'Relaxed', 'Curious'])
                elif positive_events > negative_events * 2:
                    # User is enjoying content, suggest maintaining or enhancing mood
                    current_mood = self.user_preferences['current_mood']
                    if current_mood == 'Neutral':
                        return random.choice(['Happy', 'Energetic', 'Curious'])
        
        # Default: no suggestion
        return self.user_preferences['current_mood']
