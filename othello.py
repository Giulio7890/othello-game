import tkinter as tk
from tkinter import messagebox, ttk, filedialog, colorchooser
import math
import random
import time
import json
import os
from datetime import datetime
import pygame
from PIL import Image, ImageTk
import threading
import pickle
import shutil

class SoundManager:
    def __init__(self):
        self.enabled = False
        self.current_theme = "default"
        self.volume = 0.5
        self.themes = {
            "default": {
                'move': 'sounds/default/move.wav',
                'capture': 'sounds/default/capture.wav',
                'game_over': 'sounds/default/game_over.wav',
                'hint': 'sounds/default/hint.wav',
                'button': 'sounds/default/button.wav',
                'victory': 'sounds/default/victory.wav',
                'defeat': 'sounds/default/defeat.wav',
                'menu_music': 'sounds/default/menu_music.wav',
                'game_music': 'sounds/default/game_music.wav',
                'error': 'sounds/default/error.wav',
                'achievement': 'sounds/default/achievement.wav',
                'countdown': 'sounds/default/countdown.wav',
                'piece_flip': 'sounds/default/piece_flip.wav',
                'menu_hover': 'sounds/default/menu_hover.wav',
                'menu_select': 'sounds/default/menu_select.wav'
            },
            "classic": {
                'move': 'sounds/classic/move.wav',
                'capture': 'sounds/classic/capture.wav',
                'game_over': 'sounds/classic/game_over.wav',
                'hint': 'sounds/classic/hint.wav',
                'button': 'sounds/classic/button.wav',
                'victory': 'sounds/classic/victory.wav',
                'defeat': 'sounds/classic/defeat.wav',
                'menu_music': 'sounds/classic/menu_music.wav',
                'game_music': 'sounds/classic/game_music.wav',
                'error': 'sounds/classic/error.wav',
                'achievement': 'sounds/classic/achievement.wav',
                'countdown': 'sounds/classic/countdown.wav',
                'piece_flip': 'sounds/classic/piece_flip.wav',
                'menu_hover': 'sounds/classic/menu_hover.wav',
                'menu_select': 'sounds/classic/menu_select.wav'
            },
            "modern": {
                'move': 'sounds/modern/move.wav',
                'capture': 'sounds/modern/capture.wav',
                'game_over': 'sounds/modern/game_over.wav',
                'hint': 'sounds/modern/hint.wav',
                'button': 'sounds/modern/button.wav',
                'victory': 'sounds/modern/victory.wav',
                'defeat': 'sounds/modern/defeat.wav',
                'menu_music': 'sounds/modern/menu_music.wav',
                'game_music': 'sounds/modern/game_music.wav',
                'error': 'sounds/modern/error.wav',
                'achievement': 'sounds/modern/achievement.wav',
                'countdown': 'sounds/modern/countdown.wav',
                'piece_flip': 'sounds/modern/piece_flip.wav',
                'menu_hover': 'sounds/modern/menu_hover.wav',
                'menu_select': 'sounds/modern/menu_select.wav'
            }
        }
        
        self.music_channel = None
        self.effect_channels = []
        self.MAX_EFFECT_CHANNELS = 8
        
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(self.MAX_EFFECT_CHANNELS + 1)  # +1 for music
            self.music_channel = pygame.mixer.Channel(0)
            for i in range(1, self.MAX_EFFECT_CHANNELS + 1):
                self.effect_channels.append(pygame.mixer.Channel(i))
            self.enabled = True
            self.sounds = {}
            self.load_sound_theme(self.current_theme)
        except:
            print("Warning: Sound system initialization failed")
            pass

    def load_sound_theme(self, theme_name):
        if theme_name in self.themes:
            self.sounds = {}
            theme = self.themes[theme_name]
            for name, file in theme.items():
                if os.path.exists(file):
                    try:
                        self.sounds[name] = pygame.mixer.Sound(file)
                        self.sounds[name].set_volume(self.volume)
                    except:
                        print(f"Warning: Failed to load sound {file}")
            self.current_theme = theme_name
            return True
        return False

    def play(self, sound_name, loop=0):
        if not self.enabled or sound_name not in self.sounds:
            return
            
        try:
            if sound_name in ['menu_music', 'game_music']:
                # Music goes to dedicated channel
                self.music_channel.stop()
                self.music_channel.play(self.sounds[sound_name], loops=loop)
            else:
                # Find free channel for effects
                for channel in self.effect_channels:
                    if not channel.get_busy():
                        channel.play(self.sounds[sound_name], loops=loop)
                        break
        except:
            pass

    def stop_music(self):
        if self.enabled and self.music_channel:
            try:
                self.music_channel.stop()
            except:
                pass

    def fade_out_music(self, time_ms=1000):
        if self.enabled and self.music_channel:
            try:
                self.music_channel.fadeout(time_ms)
            except:
                pass

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
        if self.enabled:
            try:
                for sound in self.sounds.values():
                    sound.set_volume(self.volume)
            except:
                pass

    def get_available_themes(self):
        return list(self.themes.keys())

    def play_sequence(self, sound_names, delay_ms=500):
        """Play a sequence of sounds with delay between them"""
        if not self.enabled:
            return
            
        def play_next(sequence):
            if sequence:
                sound_name = sequence.pop(0)
                self.play(sound_name)
                if sequence:
                    self.root.after(delay_ms, lambda: play_next(sequence))
        
        play_next(sound_names.copy())

    def is_music_playing(self):
        return self.enabled and self.music_channel and self.music_channel.get_busy()

    def set_music_volume(self, volume):
        if self.enabled and self.music_channel:
            try:
                self.music_channel.set_volume(max(0.0, min(1.0, volume)))
            except:
                pass

    def set_effects_volume(self, volume):
        if self.enabled:
            try:
                for channel in self.effect_channels:
                    channel.set_volume(max(0.0, min(1.0, volume)))
            except:
                pass

class Achievement:
    def __init__(self):
        self.achievements = {
            'first_win': {
                'name': 'First Victory',
                'description': 'Win your first game',
                'unlocked': False,
                'icon': '🏆'
            },
            'master_strategist': {
                'name': 'Master Strategist',
                'description': 'Win 5 games in a row',
                'unlocked': False,
                'icon': '🎯'
            },
            'speed_demon': {
                'name': 'Speed Demon',
                'description': 'Win a game in less than 2 minutes',
                'unlocked': False,
                'icon': '⚡'
            },
            'corner_master': {
                'name': 'Corner Master',
                'description': 'Capture all four corners in a game',
                'unlocked': False,
                'icon': '📍'
            },
            'perfect_game': {
                'name': 'Perfect Game',
                'description': 'Win with more than 75% of the pieces',
                'unlocked': False,
                'icon': '💯'
            }
        }
        self.load_achievements()

    def load_achievements(self):
        try:
            with open('achievements.json', 'r') as f:
                saved = json.load(f)
                for key, value in saved.items():
                    if key in self.achievements:
                        self.achievements[key]['unlocked'] = value['unlocked']
        except:
            pass

    def save_achievements(self):
        try:
            with open('achievements.json', 'w') as f:
                json.dump(self.achievements, f)
        except:
            pass

    def unlock(self, achievement_id):
        if achievement_id in self.achievements and not self.achievements[achievement_id]['unlocked']:
            self.achievements[achievement_id]['unlocked'] = True
            self.save_achievements()
            return self.achievements[achievement_id]
        return None

    def check_game_achievements(self, game_state):
        unlocked = []
        
        # Check first win
        if not self.achievements['first_win']['unlocked'] and game_state['winner'] == 'player':
            unlocked.append(self.unlock('first_win'))
        
        # Check win streak
        if not self.achievements['master_strategist']['unlocked'] and game_state['win_streak'] >= 5:
            unlocked.append(self.unlock('master_strategist'))
        
        # Check speed
        if not self.achievements['speed_demon']['unlocked'] and \
           game_state['winner'] == 'player' and game_state['game_time'] < 120:
            unlocked.append(self.unlock('speed_demon'))
        
        # Check corners
        if not self.achievements['corner_master']['unlocked'] and \
           game_state['corners_owned'] == 4:
            unlocked.append(self.unlock('corner_master'))
        
        # Check perfect game
        if not self.achievements['perfect_game']['unlocked'] and \
           game_state['winner'] == 'player' and \
           (game_state['player_pieces'] / (game_state['player_pieces'] + game_state['bot_pieces'])) > 0.75:
            unlocked.append(self.unlock('perfect_game'))
        
        return unlocked

class GameSaver:
    def __init__(self):
        self.save_dir = 'saves'
        os.makedirs(self.save_dir, exist_ok=True)

    def save_game(self, game_state, filename=None):
        if filename is None:
            filename = f"othello_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sav"
        
        filepath = os.path.join(self.save_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(game_state, f)
            return True
        except:
            return False

    def load_game(self, filename):
        filepath = os.path.join(self.save_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def get_save_files(self):
        try:
            return [f for f in os.listdir(self.save_dir) if f.endswith('.sav')]
        except:
            return []

    def delete_save(self, filename):
        filepath = os.path.join(self.save_dir, filename)
        try:
            os.remove(filepath)
            return True
        except:
            return False

class GameReplay:
    def __init__(self):
        self.replay_dir = 'replays'
        os.makedirs(self.replay_dir, exist_ok=True)
        self.current_replay = None
        self.replay_speed = 1.0
        self.is_playing = False
        self.replay_paused = False

    def save_replay(self, move_history, game_info, filename=None):
        if filename is None:
            filename = f"othello_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rep"
        
        filepath = os.path.join(self.replay_dir, filename)
        replay_data = {
            'moves': move_history,
            'info': game_info,
            'timestamp': datetime.now().isoformat(),
            'analysis': [],  # Store analysis for each move
            'chat_history': [],  # Store chat messages during the game
            'statistics': {
                'average_move_time': 0,
                'pieces_flipped': 0,
                'corners_captured': 0,
                'critical_moves': []
            }
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(replay_data, f)
            return True
        except:
            return False

    def load_replay(self, filename):
        filepath = os.path.join(self.replay_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                self.current_replay = pickle.load(f)
                return self.current_replay
        except:
            return None

    def get_replay_files(self):
        try:
            return [f for f in os.listdir(self.replay_dir) if f.endswith('.rep')]
        except:
            return []

    def delete_replay(self, filename):
        filepath = os.path.join(self.replay_dir, filename)
        try:
            os.remove(filepath)
            return True
        except:
            return False

    def start_replay(self, board, speed=1.0):
        if not self.current_replay:
            return False
        self.replay_speed = speed
        self.is_playing = True
        self.replay_paused = False
        return True

    def pause_replay(self):
        self.replay_paused = True

    def resume_replay(self):
        self.replay_paused = False

    def stop_replay(self):
        self.is_playing = False
        self.replay_paused = False

    def set_replay_speed(self, speed):
        """Set replay speed (0.5 = half speed, 2.0 = double speed)"""
        self.replay_speed = max(0.1, min(5.0, speed))

    def get_next_move(self):
        """Get next move in the replay sequence"""
        if not self.current_replay or not self.is_playing or self.replay_paused:
            return None
        if not self.current_replay['moves']:
            self.stop_replay()
            return None
        return self.current_replay['moves'].pop(0)

    def export_replay_to_gif(self, filename):
        """Export replay as animated GIF"""
        # Implementation for GIF export would go here
        pass

    def add_comment_to_replay(self, move_index, comment):
        """Add a comment to a specific move in the replay"""
        if self.current_replay and 0 <= move_index < len(self.current_replay['moves']):
            if 'comments' not in self.current_replay['moves'][move_index]:
                self.current_replay['moves'][move_index]['comments'] = []
            self.current_replay['moves'][move_index]['comments'].append({
                'text': comment,
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False

    def analyze_replay(self):
        """Analyze the replay for interesting patterns and statistics"""
        if not self.current_replay:
            return None

        analysis = {
            'total_moves': len(self.current_replay['moves']),
            'average_time_per_move': 0,
            'longest_move': 0,
            'most_pieces_flipped': 0,
            'critical_moves': [],
            'player_stats': {
                'black': {'total_time': 0, 'pieces_captured': 0},
                'white': {'total_time': 0, 'pieces_captured': 0}
            }
        }

        # Analyze moves
        for move in self.current_replay['moves']:
            # Add analysis logic here
            pass

        return analysis

    def create_replay_summary(self):
        """Create a text summary of the replay"""
        if not self.current_replay:
            return "No replay loaded"

        summary = []
        summary.append("Game Replay Summary")
        summary.append("=" * 20)
        summary.append(f"Date: {self.current_replay['timestamp']}")
        summary.append(f"Total Moves: {len(self.current_replay['moves'])}")
        
        if 'info' in self.current_replay:
            info = self.current_replay['info']
            summary.append(f"Winner: {info.get('winner', 'Unknown')}")
            summary.append(f"Final Score: {info.get('final_score', 'Unknown')}")
            
        return "\n".join(summary)

class GameAnalyzer:
    """Analyzes game state and provides insights"""
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_position(self, board, current_player):
        """Analyzes current board position and returns insights"""
        board_str = ''.join(''.join(row) for row in board)
        if board_str in self.analysis_cache:
            return self.analysis_cache[board_str]
            
        analysis = {
            'control': self._analyze_board_control(board),
            'mobility': self._analyze_mobility(board, current_player),
            'stability': self._analyze_stability(board),
            'potential': self._analyze_potential(board, current_player)
        }
        
        self.analysis_cache[board_str] = analysis
        return analysis
    
    def _analyze_board_control(self, board):
        """Analyzes which player controls different areas of the board"""
        corners = [(0,0), (0,7), (7,0), (7,7)]
        edges = [(i,0) for i in range(1,7)] + [(i,7) for i in range(1,7)] + \
                [(0,i) for i in range(1,7)] + [(7,i) for i in range(1,7)]
        center = [(i,j) for i in range(3,5) for j in range(3,5)]
        
        control = {
            'corners': {'B': 0, 'W': 0},
            'edges': {'B': 0, 'W': 0},
            'center': {'B': 0, 'W': 0}
        }
        
        for x, y in corners:
            if board[x][y] == 'B': control['corners']['B'] += 1
            elif board[x][y] == 'W': control['corners']['W'] += 1
                
        for x, y in edges:
            if board[x][y] == 'B': control['edges']['B'] += 1
            elif board[x][y] == 'W': control['edges']['W'] += 1
                
        for x, y in center:
            if board[x][y] == 'B': control['center']['B'] += 1
            elif board[x][y] == 'W': control['center']['W'] += 1
                
        return control
    
    def _analyze_mobility(self, board, current_player):
        """Analyzes movement possibilities for both players"""
        moves_black = sum(1 for i in range(8) for j in range(8) 
                         if self._is_valid_move(board, i, j, 'B'))
        moves_white = sum(1 for i in range(8) for j in range(8) 
                         if self._is_valid_move(board, i, j, 'W'))
        
        return {
            'B': moves_black,
            'W': moves_white,
            'ratio': moves_black / max(moves_white, 1)
        }
    
    def _analyze_stability(self, board):
        """Analyzes how stable/permanent each player's pieces are"""
        stable = {'B': 0, 'W': 0}
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        for i in range(8):
            for j in range(8):
                if board[i][j] != ' ':
                    is_stable = True
                    for dx, dy in directions:
                        if not self._check_line_stability(board, i, j, dx, dy):
                            is_stable = False
                            break
                    if is_stable:
                        stable[board[i][j]] += 1
        
        return stable
    
    def _analyze_potential(self, board, current_player):
        """Analyzes potential future moves and territory"""
        opponent = 'W' if current_player == 'B' else 'B'
        potential = {
            'territory': {'B': 0, 'W': 0},
            'frontier': {'B': 0, 'W': 0}
        }
        
        for i in range(8):
            for j in range(8):
                if board[i][j] == ' ':
                    if self._is_valid_move(board, i, j, current_player):
                        potential['territory'][current_player] += 1
                    if self._is_valid_move(board, i, j, opponent):
                        potential['territory'][opponent] += 1
                elif board[i][j] in ['B', 'W']:  # Verifica che il valore sia valido
                    # Count frontier pieces (pieces adjacent to empty spaces)
                    if self._has_empty_neighbor(board, i, j):
                        potential['frontier'][board[i][j]] += 1
        
        return potential
    
    def _is_valid_move(self, board, row, col, player):
        """Checks if a move is valid"""
        if board[row][col] != ' ':
            return False
            
        opponent = 'W' if player == 'B' else 'B'
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if not (0 <= r < 8 and 0 <= c < 8):
                continue
            if board[r][c] != opponent:
                continue
                
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
                return True
        return False
    
    def _check_line_stability(self, board, row, col, dx, dy):
        """Checks if a line of pieces is stable"""
        player = board[row][col]
        r, c = row + dx, col + dy
        count = 1
        
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
            count += 1
            r += dx
            c += dy
            
        r, c = row - dx, col - dy
        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
            count += 1
            r -= dx
            c -= dy
            
        return count >= 4
    
    def _has_empty_neighbor(self, board, row, col):
        """Checks if a position has any empty neighboring squares"""
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == ' ':
                    return True
        return False

class OthelloBot:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.move_history = []
        self.stats = self.load_stats()
        self.personality = self.generate_personality()
        
        # Evaluation weights for different positions
        self.POSITION_WEIGHTS = [
            [100, -20, 10,  5,  5, 10, -20, 100],
            [-20, -50,  -2, -2, -2, -2, -50, -20],
            [10,   -2,   8,  1,  1,  8,  -2,  10],
            [5,    -2,   1,  1,  1,  1,  -2,   5],
            [5,    -2,   1,  1,  1,  1,  -2,   5],
            [10,   -2,   8,  1,  1,  8,  -2,  10],
            [-20, -50,  -2, -2, -2, -2, -50, -20],
            [100, -20, 10,  5,  5, 10, -20, 100]
        ]
        
        # Cache for evaluated positions
        self.position_cache = {}

    def generate_personality(self):
        personalities = {
            'easy': {
                'name': 'Novice',
                'style': 'Aggressive',
                'description': 'Makes random moves but tries to capture pieces when possible.'
            },
            'medium': {
                'name': 'Strategist',
                'style': 'Balanced',
                'description': 'Focuses on corners and edges while maintaining board control.'
            },
            'hard': {
                'name': 'Master',
                'style': 'Defensive',
                'description': 'Uses advanced strategies and position evaluation.'
            },
            'impossible': {
                'name': 'Grandmaster',
                'style': 'Perfect',
                'description': 'Unbeatable AI with deep look-ahead and perfect strategy.'
            }
        }
        return personalities[self.difficulty]

    def load_stats(self):
        try:
            with open('othello_stats.json', 'r') as f:
                stats = json.load(f)
                if self.difficulty not in stats['difficulty_stats']:
                    stats['difficulty_stats'][self.difficulty] = {
                        'games_played': 0,
                        'wins': 0,
                        'losses': 0,
                        'best_time': float('inf'),
                        'total_moves': 0,
                        'average_score': 0,
                        'win_streak': 0,
                        'best_win_streak': 0
                    }
                return stats
        except:
            return {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'best_time': float('inf'),
                'total_moves': 0,
                'difficulty_stats': {
                    self.difficulty: {
                        'games_played': 0,
                        'wins': 0,
                        'losses': 0,
                        'best_time': float('inf'),
                        'total_moves': 0,
                        'average_score': 0,
                        'win_streak': 0,
                        'best_win_streak': 0
                    }
                }
            }

    def save_stats(self):
        with open('othello_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=4)

    def get_hint(self, board, current_player):
        return self.get_minimax_move(board, current_player, 3)  # Depth 3 for hints

    def get_impossible_move(self, board, valid_moves, current_player):
        return self.get_minimax_move(board, current_player, 8)  # Depth 8 for impossible mode

    def get_minimax_move(self, board, current_player, depth):
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(board, i, j, current_player):
                    valid_moves.append((i, j))
        
        if not valid_moves:
            return None
            
        # Prioritize corners if available
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for move in valid_moves:
            if move in corners:
                return move
        
        # Apply minimax with alpha-beta pruning
        for move in valid_moves:
            board_copy = [row[:] for row in board]
            self.make_move(board_copy, move[0], move[1], current_player)
            score = self.minimax(board_copy, depth - 1, alpha, beta, False, current_player)
            
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            
            if beta <= alpha:
                break
        
        return best_move

    def board_to_string(self, board):
        return ''.join(''.join(row) for row in board)

    def minimax(self, board, depth, alpha, beta, is_maximizing, original_player):
        board_str = self.board_to_string(board)
        cache_key = (board_str, depth, is_maximizing, original_player)
        
        if cache_key in self.position_cache:
            return self.position_cache[cache_key]
            
        if depth == 0:
            score = self.evaluate_position(board, original_player)
            self.position_cache[cache_key] = score
            return score
        
        current = original_player if is_maximizing else ('W' if original_player == 'B' else 'B')
        valid_moves = []
        
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(board, i, j, current):
                    valid_moves.append((i, j))
        
        if not valid_moves:
            score = self.evaluate_position(board, original_player)
            self.position_cache[cache_key] = score
            return score
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                board_copy = [row[:] for row in board]
                self.make_move(board_copy, move[0], move[1], current)
                eval = self.minimax(board_copy, depth - 1, alpha, beta, False, original_player)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.position_cache[cache_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                board_copy = [row[:] for row in board]
                self.make_move(board_copy, move[0], move[1], current)
                eval = self.minimax(board_copy, depth - 1, alpha, beta, True, original_player)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.position_cache[cache_key] = min_eval
            return min_eval

    def evaluate_position(self, board, player):
        opponent = 'W' if player == 'B' else 'B'
        score = 0
        
        # Count pieces with position weights
        for i in range(8):
            for j in range(8):
                if board[i][j] == player:
                    score += self.POSITION_WEIGHTS[i][j]
                elif board[i][j] == opponent:
                    score -= self.POSITION_WEIGHTS[i][j]
        
        # Corner control (most important)
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for x, y in corners:
            if board[x][y] == player:
                score += 25
            elif board[x][y] == opponent:
                score -= 25
        
        return score

    def get_move(self, board, current_player):
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(board, i, j, current_player):
                    valid_moves.append((i, j))
        
        if not valid_moves:
            return None

        if self.difficulty == 'easy':
            return random.choice(valid_moves)
        elif self.difficulty == 'medium':
            return self.get_minimax_move(board, current_player, 2)  # Depth 2 for medium
        elif self.difficulty == 'hard':
            return self.get_minimax_move(board, current_player, 3)  # Depth 3 for hard
        else:  # impossible
            return self.get_minimax_move(board, current_player, 4)  # Depth 4 for impossible

    def is_valid_move(self, board, row, col, current_player):
        if not (0 <= row < 8 and 0 <= col < 8):
            return False
        if board[row][col] != ' ':
            return False

        opponent = 'W' if current_player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if not (0 <= r < 8 and 0 <= c < 8):
                continue
            if board[r][c] != opponent:
                continue
            
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == current_player:
                return True
        return False

    def make_move(self, board, row, col, current_player):
        """Makes a move on the board and returns True if successful"""
        opponent = 'W' if current_player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        board[row][col] = current_player
        pieces_flipped = False
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            
            while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
            
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == current_player:
                for flip_r, flip_c in to_flip:
                    board[flip_r][flip_c] = current_player
                pieces_flipped = True
        
        if not pieces_flipped:
            board[row][col] = ' '
            
        return pieces_flipped

    def evaluate_board(self, board, current_player):
        # Corner positions are worth more
        corner_value = 5
        edge_value = 3
        normal_value = 1
        
        score = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edges = [(i, 0) for i in range(1, 7)] + [(i, 7) for i in range(1, 7)] + \
                [(0, i) for i in range(1, 7)] + [(7, i) for i in range(1, 7)]
        
        # Count pieces and evaluate positions
        for i in range(8):
            for j in range(8):
                if board[i][j] == current_player:
                    if (i, j) in corners:
                        score += corner_value
                    elif (i, j) in edges:
                        score += edge_value
                    else:
                        score += normal_value
                elif board[i][j] != ' ':
                    if (i, j) in corners:
                        score -= corner_value
                    elif (i, j) in edges:
                        score -= edge_value
                    else:
                        score -= normal_value
        
        # Add mobility factor (number of valid moves)
        mobility_score = 0
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(board, i, j, current_player):
                    mobility_score += 1
                if self.is_valid_move(board, i, j, 'W' if current_player == 'B' else 'B'):
                    mobility_score -= 1
        
        return score + (mobility_score * 0.5)  # Add mobility score with weight 0.5

class GameStats:
    """Tracks and manages game statistics"""
    def __init__(self):
        self.stats = self.load_stats()
        
    def load_stats(self):
        try:
            with open('game_stats.json', 'r') as f:
                return json.load(f)
        except:
            return {
                'games': [],
                'achievements': {},
                'records': {
                    'fastest_win': float('inf'),
                    'biggest_margin': 0,
                    'longest_streak': 0,
                    'most_pieces': 0
                },
                'totals': {
                    'games_played': 0,
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'total_time': 0,
                    'total_moves': 0
                }
            }
    
    def save_stats(self):
        with open('game_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=4)
    
    def add_game(self, game_data):
        """Adds a new game to statistics and updates records"""
        self.stats['games'].append(game_data)
        self.stats['totals']['games_played'] += 1
        
        if game_data['winner'] == 'player':
            self.stats['totals']['wins'] += 1
            if game_data['time'] < self.stats['records']['fastest_win']:
                self.stats['records']['fastest_win'] = game_data['time']
        elif game_data['winner'] == 'bot':
            self.stats['totals']['losses'] += 1
        else:
            self.stats['totals']['draws'] += 1
            
        self.stats['totals']['total_time'] += game_data['time']
        self.stats['totals']['total_moves'] += len(game_data['moves'])
        
        # Update records
        margin = abs(game_data['final_score']['player'] - game_data['final_score']['bot'])
        if margin > self.stats['records']['biggest_margin']:
            self.stats['records']['biggest_margin'] = margin
            
        if game_data['winner'] == 'player':
            streak = 1
            for game in reversed(self.stats['games'][:-1]):
                if game['winner'] == 'player':
                    streak += 1
                else:
                    break
            if streak > self.stats['records']['longest_streak']:
                self.stats['records']['longest_streak'] = streak
                
        max_pieces = max(game_data['final_score']['player'], 
                        game_data['final_score']['bot'])
        if max_pieces > self.stats['records']['most_pieces']:
            self.stats['records']['most_pieces'] = max_pieces
            
        self.save_stats()
    
    def get_summary(self):
        """Returns a formatted summary of statistics"""
        totals = self.stats['totals']
        records = self.stats['records']
        
        if totals['games_played'] == 0:
            return "No games played yet!"
            
        win_rate = (totals['wins'] / totals['games_played']) * 100
        avg_time = totals['total_time'] / totals['games_played']
        avg_moves = totals['total_moves'] / totals['games_played']
        
        summary = [
            f"Games Played: {totals['games_played']}",
            f"Win Rate: {win_rate:.1f}%",
            f"Average Game Time: {avg_time:.1f}s",
            f"Average Moves per Game: {avg_moves:.1f}",
            "\nRecords:",
            f"Fastest Win: {records['fastest_win']:.1f}s",
            f"Biggest Margin: {records['biggest_margin']} pieces",
            f"Longest Win Streak: {records['longest_streak']} games",
            f"Most Pieces in Game: {records['most_pieces']}"
        ]
        
        return "\n".join(summary)

class Othello:
    def __init__(self, root, bot_difficulty, move_time=30, reset_timer=True):
        self.root = root
        self.root.title("Othello Game")
        
        # Flag to track if the game is running
        self.running = True
        
        # Store move time and reset option
        self.MOVE_TIME = move_time
        self.should_reset_timer = reset_timer
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize sound manager
        self.sound_manager = SoundManager()
        
        
        # Center the window
        window_width = 1200  # Increased from 1000
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Constants
        self.BOARD_SIZE = 8
        self.CELL_SIZE = 60
        self.PIECE_RADIUS = 25
        self.ANIMATION_SPEED = 10  # milliseconds
        
        # Colors
        self.BOARD_COLOR = "#1B5E20"
        self.LINE_COLOR = "#000000"
        self.BLACK = "#000000"
        self.WHITE = "#FFFFFF"
        self.VALID_MOVE = "#4CAF50"
        self.HINT_COLOR = "#FFD700"
        self.SELECTED_COLOR = "#FF4500"
        self.BORDER_COLOR = "#8B4513"  # Wooden brown color
        self.STRATEGIC_POINT_COLOR = "#000000"  # Changed to black
        
        # Strategic points coordinates (row, col) - positioned at grid intersections
        self.strategic_points = [
            (2, 2), (2, 6),  # Row 2 points
            (6, 2), (6, 6)   # Row 6 points
        ]
        
        # Game state
        self.board = [[' ' for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.current_player = 'B'
        self.bot = OthelloBot(bot_difficulty)
        self.is_player_turn = True
        self.move_history = []
        self.start_time = time.time()
        self.player_time = self.MOVE_TIME
        self.bot_time = self.MOVE_TIME
        self.is_paused = False
        self.hint_move = None
        self.selected_move = None
        self.animation_queue = []
        self.is_animating = False
        
        # Add new components
        self.analyzer = GameAnalyzer()
        self.stats = GameStats()
        
        # Initialize game state
        self.game_start_time = time.time()
        self.analysis_history = []
        
        # Create main game frame
        self.game_frame = tk.Frame(root)
        self.game_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Create top frame for timers and controls
        self.top_frame = tk.Frame(self.game_frame)
        self.top_frame.pack(fill='x', pady=(0, 20))
        
        # Create timer labels with progress bars
        self.player_timer_frame = tk.Frame(self.top_frame)
        self.player_timer_frame.pack(side='left', padx=10)
        
        self.player_timer = tk.Label(
            self.player_timer_frame,
            text=f"Your time: {self.player_time}s",
            font=("Arial", 12, "bold")
        )
        self.player_timer.pack()
        
        self.player_progress = ttk.Progressbar(
            self.player_timer_frame,
            length=200,
            mode='determinate',
            maximum=self.MOVE_TIME
        )
        self.player_progress.pack()
        
        self.bot_timer_frame = tk.Frame(self.top_frame)
        self.bot_timer_frame.pack(side='right', padx=10)
        
        self.bot_timer = tk.Label(
            self.bot_timer_frame,
            text=f"Bot time: {self.bot_time}s",
            font=("Arial", 12, "bold")
        )
        self.bot_timer.pack()
        
        self.bot_progress = ttk.Progressbar(
            self.bot_timer_frame,
            length=200,
            mode='determinate',
            maximum=self.MOVE_TIME
        )
        self.bot_progress.pack()
        
        # Create control buttons frame
        self.control_frame = tk.Frame(self.top_frame)
        self.control_frame.pack(expand=True, fill='x', padx=10)
        
        # Create left side buttons frame
        self.left_buttons = tk.Frame(self.control_frame)
        self.left_buttons.pack(side='left', expand=True)
        
        # Create right side buttons frame
        self.right_buttons = tk.Frame(self.control_frame)
        self.right_buttons.pack(side='right', expand=True)
        
        # Create volume control
        self.volume_frame = tk.Frame(self.left_buttons)
        self.volume_frame.pack(side='left', padx=5)
        
        self.volume_label = tk.Label(
            self.volume_frame,
            text="Volume:",
            font=("Arial", 10)
        )
        self.volume_label.pack(side='left')
        
        self.volume_scale = ttk.Scale(
            self.volume_frame,
            from_=0,
            to=100,
            orient='horizontal',
            length=100,
            command=self.update_volume
        )
        self.volume_scale.set(50)
        self.volume_scale.pack(side='left')
        
        # Create game control buttons in the right frame
        button_configs = [
            ("Hint", self.show_hint),
            ("Undo", self.undo_move),
            ("Pause", self.toggle_pause),
            ("Analysis", self.show_analysis),
            ("Menu", self.show_main_menu)  # Changed from "Main Menu" to "Menu"
        ]
        
        for button_info in button_configs:
            btn = tk.Button(
                self.right_buttons,
                text=button_info[0],
                command=button_info[1],
                font=("Arial", 10),
                bg='#4CAF50',
                fg='white',
                padx=10,
                pady=5
            )
            btn.pack(side='left', padx=5)
            
            # Store button references
            if button_info[0] == "Pause":
                self.pause_button = btn
            elif button_info[0] == "Hint":
                self.hint_button = btn
            elif button_info[0] == "Undo":
                self.undo_button = btn
            elif button_info[0] == "Analysis":
                self.analysis_button = btn
            elif button_info[0] == "Menu":  # Changed from "Main Menu" to "Menu"
                self.menu_button = btn
        
        # Create game area frame
        self.game_area_frame = tk.Frame(self.game_frame)
        self.game_area_frame.pack(expand=True, fill='both')
        
        # Create canvas with adjusted size for padding
        canvas_size = (self.BOARD_SIZE * self.CELL_SIZE) + 60  # Add padding on both sides
        self.canvas = tk.Canvas(
            self.game_area_frame, 
            width=canvas_size,
            height=canvas_size,
            bg=self.BOARD_COLOR,
            highlightthickness=0
        )
        self.canvas.pack(side='left', padx=(0, 20))
        
        # Create right panel frame
        self.right_panel = tk.Frame(self.game_area_frame)
        self.right_panel.pack(side='right', fill='both', expand=True)
        
        # Create move history frame
        self.create_move_history_frame()
        
        # Create statistics frame
        self.stats_frame = tk.LabelFrame(
            self.right_panel,
            text="Statistics",
            font=("Arial", 12, "bold")
        )
        self.stats_frame.pack(fill='both', expand=True)
        
        # Create statistics labels
        self.stats_text = tk.Text(
            self.stats_frame,
            width=30,
            height=10,
            font=("Arial", 10),
            wrap=tk.WORD,
            state='disabled'  # Make it read-only
        )
        self.stats_text.pack(fill='both', expand=True)
        self.update_stats()
        
        # Create status label
        self.status_label = tk.Label(
            root,
            text="Your turn (Black)",
            font=("Arial", 14, "bold"),
            pady=10
        )
        self.status_label.pack()
        
        # Bind events
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_hover)
        
        # Initialize the board
        self.initialize_board()
        self.draw_board()
        self.update_valid_moves()
        
        # Start timer
        self.update_timer()
        
        # Add this line to track the last move
        self.last_move = None

    def update_volume(self, value):
        volume = float(value) / 100
        self.sound_manager.set_volume(volume)

    def handle_hover(self, event):
        if not self.is_player_turn or self.is_paused:
            return
            
        padding = 30
        board_x = event.x - padding
        board_y = event.y - padding
        
        # Check if hover is within the board area
        if (0 <= board_x <= self.BOARD_SIZE * self.CELL_SIZE and 
            0 <= board_y <= self.BOARD_SIZE * self.CELL_SIZE):
            
            col = board_x // self.CELL_SIZE
            row = board_y // self.CELL_SIZE
            
            if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
                if self.is_valid_move(row, col):
                    self.selected_move = (row, col)
                    self.draw_board()
                    self.update_valid_moves()
                    
                    # Draw selected move indicator
                    x = padding + col * self.CELL_SIZE + self.CELL_SIZE // 2
                    y = padding + row * self.CELL_SIZE + self.CELL_SIZE // 2
                    self.canvas.create_oval(
                        x - 8, y - 8,
                        x + 8, y + 8,
                        fill=self.SELECTED_COLOR,
                        outline=self.LINE_COLOR,
                        tags="selected"
                    )
                else:
                    self.selected_move = None
                    self.draw_board()
                    self.update_valid_moves()
        else:
            self.selected_move = None
            self.draw_board()
            self.update_valid_moves()

    def add_to_history(self, move, player):
        """Add a move to the history"""
        # Standardize the move format
        move_data = {
            'position': move,
            'player': player,
            'time': time.time() - self.game_start_time
        }
        
        # Add move to history
        self.move_history.append(move_data)
        
        # Update display with proper formatting
        try:
            self.history_text.configure(state='normal')
            row, col = move
            move_text = f"{len(self.move_history)}. {player}: ({row + 1}, {col + 1})\n"
            self.history_text.insert(tk.END, move_text)
            self.history_text.see(tk.END)
            self.history_text.configure(state='disabled')
            self.history_text.update()  # Force update
        except Exception as e:
            print(f"Error updating move history: {e}")

    def create_move_history_frame(self):
        """Create the move history frame"""
        # Create move history frame
        self.history_frame = tk.LabelFrame(
            self.right_panel,
            text="Move History",
            font=("Arial", 12, "bold")
        )
        self.history_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Create move history text with scrollbar
        self.history_text = tk.Text(
            self.history_frame,
            width=30,
            height=15,
            font=("Arial", 10),
            wrap=tk.WORD
        )
        self.history_scroll = ttk.Scrollbar(
            self.history_frame,
            orient="vertical",
            command=self.history_text.yview
        )
        self.history_text.configure(yscrollcommand=self.history_scroll.set)
        
        # Pack the widgets
        self.history_text.pack(side='left', fill='both', expand=True)
        self.history_scroll.pack(side='right', fill='y')
        
        # Make it read-only
        self.history_text.configure(state='disabled')

    def undo_move(self):
        """Undo both player and bot moves at once"""
        # Se abbiamo meno di 2 mosse, non fare nulla
        if len(self.move_history) < 2:
            return
            
        # Rimuovi le ultime due mosse (giocatore e bot)
        moves_to_remove = 2
        self.move_history = self.move_history[:-moves_to_remove]
        
        # Resetta completamente la board
        self.board = [[' ' for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.initialize_board()
        
        # Pulisci tutto
        self.hint_move = None
        self.selected_move = None
        self.last_move = None
        self.canvas.delete("all")
        
        # Aggiorna la storia visualizzata
        try:
            self.history_text.configure(state='normal')
            self.history_text.delete('1.0', tk.END)
            
            # Riscrive tutta la storia delle mosse
            for i, move_data in enumerate(self.move_history, 1):
                row, col = move_data['position']
                player = move_data['player']
                move_text = f"{i}. {player}: ({row + 1}, {col + 1})\n"
                self.history_text.insert(tk.END, move_text)
                
                # Applica la mossa sulla board
                actual_player = 'B' if player in ['B', 'Player'] else 'W'
                self.board[row][col] = actual_player
                
                # Gira i pezzi
                opponent = 'W' if actual_player == 'B' else 'B'
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                            (1, 1), (-1, -1), (1, -1), (-1, 1)]
                
                for dr, dc in directions:
                    r, c = row + dr, col + dc
                    to_flip = []
                    
                    while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == opponent:
                        to_flip.append((r, c))
                        r += dr
                        c += dc
                    
                    if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == actual_player:
                        for flip_r, flip_c in to_flip:
                            self.board[flip_r][flip_c] = actual_player
            
            self.history_text.see(tk.END)
            self.history_text.configure(state='disabled')
            self.history_text.update()  # Force update
        except Exception as e:
            print(f"Error updating move history: {e}")
        
        # Forza il turno al giocatore
        self.current_player = 'B'
        self.is_player_turn = True
        
        # Resetta il timer se necessario
        if self.should_reset_timer:
            self.reset_timer()
        
        # Aggiorna tutto
        self.draw_board()
        self.update_valid_moves()
        self.update_status_message()
        self.sound_manager.play('button')

    def animate_move(self, row, col, player):
        if not self.is_animating:
            self.is_animating = True
            
            # Get the pieces to flip
            pieces_to_flip = []
            opponent = 'W' if player == 'B' else 'B'
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                         (1, 1), (-1, -1), (1, -1), (-1, 1)]
            
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if not (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE):
                    continue
                if self.board[r][c] != opponent:
                    continue
                
                to_flip = []
                while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == opponent:
                    to_flip.append((r, c))
                    r += dr
                    c += dc
                if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == player:
                    pieces_to_flip.extend(to_flip)
            
            # Animate the move
            self.animate_piece(row, col, player, pieces_to_flip)

    def animate_piece(self, row, col, player, pieces_to_flip, step=0):
        if step == 0:
            # Place the new piece
            self.board[row][col] = player
            self.draw_board()
            self.sound_manager.play('move')
            self.root.after(self.ANIMATION_SPEED, 
                          lambda: self.animate_piece(row, col, player, pieces_to_flip, step + 1))
        elif step <= len(pieces_to_flip):
            # Flip pieces one by one
            if step > 0 and step <= len(pieces_to_flip):
                r, c = pieces_to_flip[step - 1]
                self.board[r][c] = player
                self.draw_board()
                self.sound_manager.play('capture')
            if step < len(pieces_to_flip):
                self.root.after(self.ANIMATION_SPEED,
                              lambda: self.animate_piece(row, col, player, pieces_to_flip, step + 1))
            else:
                self.is_animating = False
                self.current_player = 'W' if player == 'B' else 'B'
                self.is_player_turn = not self.is_player_turn
                self.update_valid_moves()
                
                if not self.has_valid_moves():
                    self.current_player = 'W' if self.current_player == 'B' else 'B'
                    if not self.has_valid_moves():
                        self.show_game_over()

    def update_stats(self):
        stats = self.bot.stats
        difficulty_stats = stats['difficulty_stats'][self.bot.difficulty]
        
        stats_text = f"Current Bot: {self.bot.personality['name']}\n"
        stats_text += f"Style: {self.bot.personality['style']}\n"
        stats_text += f"Description: {self.bot.personality['description']}\n\n"
        stats_text += f"Games Played: {difficulty_stats['games_played']}\n"
        stats_text += f"Wins: {difficulty_stats['wins']}\n"
        stats_text += f"Losses: {difficulty_stats['losses']}\n"
        if difficulty_stats['best_time'] != float('inf'):
            stats_text += f"Best Time: {difficulty_stats['best_time']:.1f}s\n"
        stats_text += f"Total Moves: {difficulty_stats['total_moves']}\n"
        stats_text += f"Win Streak: {difficulty_stats['win_streak']}\n"
        stats_text += f"Best Win Streak: {difficulty_stats['best_win_streak']}\n"
        
        self.stats_text.configure(state='normal')  # Temporarily enable writing
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats_text)
        self.stats_text.configure(state='disabled')  # Make it read-only again

    def show_game_over(self, message=None):
        # Get final analysis
        final_analysis = self.analyzer.analyze_position(self.board, self.current_player)
        
        # Calculate scores
        black_score = sum(row.count('B') for row in self.board)
        white_score = sum(row.count('W') for row in self.board)
        
        # Prepare game data
        game_data = {
            'date': datetime.now().isoformat(),
            'difficulty': self.bot.difficulty,
            'moves': self.move_history,
            'time': time.time() - self.game_start_time,
            'final_score': {
                'player': black_score,
                'bot': white_score
            },
            'winner': 'player' if black_score > white_score else 'bot' if white_score > black_score else 'draw',
            'analysis_history': self.analysis_history,
            'final_analysis': final_analysis
        }
        
        # Update statistics
        self.stats.add_game(game_data)
        
        # Show detailed game over message
        if message is None:
            stats_summary = self.stats.get_summary()
            message = f"Game Over!\n\nFinal Score:\nYou (Black): {black_score}\nBot (White): {white_score}\n\n{stats_summary}"
            
        messagebox.showinfo("Game Over", message)
        self.show_main_menu()

    def handle_click(self, event):
        if not self.is_player_turn or self.is_paused:
            return
            
        padding = 30
        board_x = event.x - padding
        board_y = event.y - padding
        
        # Check if click is within the board area
        if (0 <= board_x <= self.BOARD_SIZE * self.CELL_SIZE and 
            0 <= board_y <= self.BOARD_SIZE * self.CELL_SIZE):
            
            col = board_x // self.CELL_SIZE
            row = board_y // self.CELL_SIZE
            
            if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
                if self.make_move(row, col):
                    # Add player's move to history
                    self.add_to_history((row, col), "Player")
                    self.update_valid_moves()
                    
                    if self.should_reset_timer:
                        self.reset_timer()
                    
                    if self.check_game_over():
                        self.show_game_over()
                        return
                    
                    self.update_status_message()
                    # Schedule bot's move
                    self.root.after(500, self.bot_move)

    def bot_move(self):
        if not self.is_player_turn and not self.is_paused:
            move = self.bot.get_move(self.board, self.current_player)
            if move:
                row, col = move
                if self.make_move(row, col):
                    # Add bot's move to history
                    self.add_to_history((row, col), "Bot")
                    self.update_valid_moves()
                    
                    if self.should_reset_timer:
                        self.reset_timer()
                    
                    if self.check_game_over():
                        self.show_game_over()
                        return
                    
                    self.update_status_message()

    def update_bot_timer(self):
        """Update the bot's timer in a separate thread"""
        while not self.is_player_turn and not self.is_paused and self.running:
            try:
                self.bot_time -= 1
                if self.bot_time <= 0:
                    self.root.after(0, self.show_game_over, "Time's up! You win!")
                    break
                
                # Update timer display using after method to ensure thread safety
                self.root.after(0, self.update_bot_timer_display)
                time.sleep(1)
            except:
                break

    def update_bot_timer_display(self):
        """Update the bot's timer display"""
        try:
            self.bot_timer.config(text=f"Bot time: {self.bot_time}s")
            self.bot_progress['value'] = self.bot_time
        except:
            pass

    def show_main_menu(self):
        self.running = False
        # Clear game
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Show main menu
        MainMenu(self.root)

    def initialize_board(self):
        # Place initial pieces in the center
        self.board[3][3] = 'W'
        self.board[3][4] = 'B'
        self.board[4][3] = 'B'
        self.board[4][4] = 'W'

    def draw_board(self):
        self.canvas.delete("all")
        
        padding = 30
        board_start_x = padding
        board_start_y = padding
        board_size = self.BOARD_SIZE * self.CELL_SIZE
        
        # Draw outer wooden border (full border)
        self.canvas.create_rectangle(
            0, 0,
            board_size + padding * 2,
            board_size + padding * 2,
            fill=self.BORDER_COLOR,
            outline=self.LINE_COLOR,
            width=2
        )
        
        # Draw board background
        self.canvas.create_rectangle(
            board_start_x,
            board_start_y,
            board_start_x + board_size,
            board_start_y + board_size,
            fill=self.BOARD_COLOR,
            outline=self.LINE_COLOR,
            width=1
        )
        
        # Draw coordinate labels
        for i in range(self.BOARD_SIZE):
            # Draw column labels (A-H)
            self.canvas.create_text(
                board_start_x + i * self.CELL_SIZE + self.CELL_SIZE//2,
                padding//2,
                text=chr(65 + i),
                font=("Arial", 12, "bold"),
                fill="white"
            )
            
            # Draw row labels (1-8)
            self.canvas.create_text(
                padding//2,
                board_start_y + i * self.CELL_SIZE + self.CELL_SIZE//2,
                text=str(i + 1),
                font=("Arial", 12, "bold"),
                fill="white"
            )
        
        # Draw grid lines
        for i in range(self.BOARD_SIZE + 1):
            # Vertical lines
            self.canvas.create_line(
                board_start_x + i * self.CELL_SIZE,
                board_start_y,
                board_start_x + i * self.CELL_SIZE,
                board_start_y + board_size,
                fill=self.LINE_COLOR,
                width=1
            )
            # Horizontal lines
            self.canvas.create_line(
                board_start_x,
                board_start_y + i * self.CELL_SIZE,
                board_start_x + board_size,
                board_start_y + i * self.CELL_SIZE,
                fill=self.LINE_COLOR,
                width=1
            )
        
        # Draw strategic points at grid intersections
        for row, col in self.strategic_points:
            x = board_start_x + col * self.CELL_SIZE
            y = board_start_y + row * self.CELL_SIZE
            # Make points slightly smaller
            self.canvas.create_oval(
                x - 3,
                y - 3,
                x + 3,
                y + 3,
                fill=self.STRATEGIC_POINT_COLOR,
                outline=self.STRATEGIC_POINT_COLOR
            )
        
        # Draw pieces
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.board[i][j] != ' ':
                    self.draw_piece(i, j, self.board[i][j], board_start_x, board_start_y)

    def draw_piece(self, row, col, color, board_start_x, board_start_y):
        x = board_start_x + col * self.CELL_SIZE + self.CELL_SIZE // 2
        y = board_start_y + row * self.CELL_SIZE + self.CELL_SIZE // 2
        
        # Draw piece shadow
        self.canvas.create_oval(
            x - self.PIECE_RADIUS + 2,
            y - self.PIECE_RADIUS + 2,
            x + self.PIECE_RADIUS + 2,
            y + self.PIECE_RADIUS + 2,
            fill="#404040"
        )
        
        # Draw piece
        piece_color = self.BLACK if color == 'B' else self.WHITE
        self.canvas.create_oval(
            x - self.PIECE_RADIUS,
            y - self.PIECE_RADIUS,
            x + self.PIECE_RADIUS,
            y + self.PIECE_RADIUS,
            fill=piece_color,
            outline=self.LINE_COLOR
        )
        
        # Add highlight for 3D effect
        if color == 'W':
            self.canvas.create_arc(
                x - self.PIECE_RADIUS,
                y - self.PIECE_RADIUS,
                x + self.PIECE_RADIUS,
                y + self.PIECE_RADIUS,
                start=0, extent=180,
                fill="#E0E0E0"
            )
        
        # Draw red dot if this is the last move
        if hasattr(self, 'last_move') and self.last_move == (row, col):
            dot_radius = self.PIECE_RADIUS // 4
            self.canvas.create_oval(
                x - dot_radius,
                y - dot_radius,
                x + dot_radius,
                y + dot_radius,
                fill="#FF0000",
                outline="#FF0000"
            )

    def draw_valid_move(self, row, col):
        padding = 30
        x = padding + col * self.CELL_SIZE + self.CELL_SIZE // 2
        y = padding + row * self.CELL_SIZE + self.CELL_SIZE // 2
        
        # Draw valid move indicator
        self.canvas.create_oval(
            x - 5, y - 5,
            x + 5, y + 5,
            fill=self.VALID_MOVE,
            outline=self.LINE_COLOR
        )

    def is_valid_move(self, row, col):
        if not (0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE):
            return False
        if self.board[row][col] != ' ':
            return False

        opponent = 'W' if self.current_player == 'B' else 'B'
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if not (0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE):
                continue
            if self.board[r][c] != opponent:
                continue
            
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == opponent:
                r += dr
                c += dc
            if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == self.current_player:
                return True
        return False

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False
            
        # Store position analysis before move
        analysis = self.analyzer.analyze_position(self.board, self.current_player)
        self.analysis_history.append(analysis)
        
        # Update last move
        self.last_move = (row, col)
        
        # Make the move
        opponent = 'W' if self.current_player == 'B' else 'B'
        self.board[row][col] = self.current_player
        pieces_flipped = False
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
            
            if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and self.board[r][c] == self.current_player:
                for flip_r, flip_c in to_flip:
                    self.board[flip_r][flip_c] = self.current_player
                pieces_flipped = True
        
        if pieces_flipped:
            move_data = {
                'player': self.current_player,
                'position': (row, col),
                'time': time.time() - self.game_start_time,
                'analysis': analysis
            }
            self.move_history.append(move_data)
            self.current_player = opponent
            self.is_player_turn = not self.is_player_turn
            return True
        
        # If no pieces were flipped, undo the move
        self.board[row][col] = ' '
        self.last_move = None
        return False

    def update_valid_moves(self):
        """Update and display valid moves on the board"""
        # Clear previous valid moves indicators
        self.canvas.delete("valid_move")
        self.canvas.delete("selected")
        self.canvas.delete("hint")
        
        self.draw_board()
        if self.is_player_turn:
            padding = 30
            for i in range(self.BOARD_SIZE):
                for j in range(self.BOARD_SIZE):
                    if self.is_valid_move(i, j):
                        x = padding + j * self.CELL_SIZE + self.CELL_SIZE // 2
                        y = padding + i * self.CELL_SIZE + self.CELL_SIZE // 2
                        self.canvas.create_oval(
                            x - 5, y - 5,
                            x + 5, y + 5,
                            fill=self.VALID_MOVE,
                            outline=self.LINE_COLOR,
                            tags="valid_move"
                        )

    def has_valid_moves(self):
        """Check if the current player has any valid moves"""
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if self.is_valid_move(i, j):
                    return True
        return False

    def get_score(self):
        black = sum(row.count('B') for row in self.board)
        white = sum(row.count('W') for row in self.board)
        return black, white

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_button.config(text="Resume" if self.is_paused else "Pause")
        if not self.is_paused and self.running:
            self.update_timer()

    def show_hint(self):
        if not self.is_player_turn or self.is_paused:
            return
            
        self.hint_move = self.bot.get_hint(self.board, self.current_player)
        if self.hint_move:
            self.draw_board()
            self.update_valid_moves()
            row, col = self.hint_move
            x = col * self.CELL_SIZE + self.CELL_SIZE // 2
            y = row * self.CELL_SIZE + self.CELL_SIZE // 2
            
            # Draw hint indicator
            self.canvas.create_oval(
                x - 8, y - 8,
                x + 8, y + 8,
                fill=self.HINT_COLOR,
                outline=self.LINE_COLOR,
                tags="hint"
            )
            
            # Remove hint after 2 seconds
            self.root.after(2000, self.remove_hint)

    def remove_hint(self):
        self.canvas.delete("hint")
        self.hint_move = None
        self.draw_board()
        self.update_valid_moves()

    def update_timer(self):
        if not self.running or self.is_paused:
            return
            
        try:
            if self.is_player_turn:
                self.player_time -= 1
                if self.player_time <= 0:
                    self.show_game_over("Time's up! Bot wins!")
                    return
                self.player_timer.config(text=f"Your time: {self.player_time}s")
                self.player_progress['value'] = self.player_time
            
            if self.running:
                self.root.after(1000, self.update_timer)
        except tk.TclError:
            # Widget was destroyed, stop the timer
            self.running = False

    def on_closing(self):
        self.running = False
        self.root.destroy()

    def reset_timer(self):
        if not self.should_reset_timer:
            return
            
        if self.is_player_turn:
            self.player_time = self.MOVE_TIME
            self.player_timer.config(text=f"Your time: {self.player_time}s")
            self.player_progress['value'] = self.MOVE_TIME
        else:
            self.bot_time = self.MOVE_TIME
            self.bot_timer.config(text=f"Bot time: {self.bot_time}s")
            self.bot_progress['value'] = self.MOVE_TIME

    def check_game_over(self):
        """
        Checks if the game is over by verifying if any player has valid moves.
        Returns True if the game is over, False otherwise.
        """
        # Save current player
        original_player = self.current_player
        
        # Check if current player has moves
        if self.has_valid_moves():
            return False
            
        # Switch player temporarily to check their moves
        self.current_player = 'W' if original_player == 'B' else 'B'
        other_has_moves = self.has_valid_moves()
        
        # Restore original player
        self.current_player = original_player
        
        # Game is over if neither player has valid moves
        return not other_has_moves

    def pass_turn(self):
        """Pass the turn to the other player when no moves are available"""
        # Before passing, check if the other player has moves
        original_player = self.current_player
        next_player = 'W' if original_player == 'B' else 'B'
        
        # Temporarily switch to check next player's moves
        self.current_player = next_player
        next_player_has_moves = self.has_valid_moves()
        self.current_player = original_player
        
        # If neither player has moves, end the game
        if not next_player_has_moves and not self.has_valid_moves():
            self.show_game_over()
            return
            
        # Otherwise, pass the turn
        self.current_player = next_player
        self.is_player_turn = not self.is_player_turn
        
        # Update the display for the next player
        self.update_valid_moves()
        self.update_status_message()
        
        # If it's bot's turn and they have valid moves, make bot move
        if not self.is_player_turn and next_player_has_moves:
            self.root.after(500, self.bot_move)

    def update_status_message(self):
        """Update the status message based on game state"""
        if self.has_valid_moves():
            if self.is_player_turn:
                self.status_label.config(text="Your turn (Black)")
            else:
                self.status_label.config(text="Bot's turn (White)")
        else:
            # Check if the other player has moves before showing pass message
            original_player = self.current_player
            next_player = 'W' if original_player == 'B' else 'B'
            
            # Temporarily switch to check next player's moves
            self.current_player = next_player
            next_player_has_moves = self.has_valid_moves()
            self.current_player = original_player
            
            if next_player_has_moves:
                if self.is_player_turn:
                    self.status_label.config(text="No valid moves available - Passing turn")
                    self.root.after(1500, self.pass_turn)
                else:
                    self.status_label.config(text="Bot has no valid moves - Passing turn")
                    self.root.after(1500, self.pass_turn)
            else:
                # If neither player has moves, end the game
                self.root.after(1000, self.show_game_over)

    def show_analysis(self):
        """Shows analysis window with current position insights"""
        if not hasattr(self, 'analysis_window') or not self.analysis_window.winfo_exists():
            self.analysis_window = tk.Toplevel(self.root)
            self.analysis_window.title("Position Analysis")
            self.analysis_window.geometry("400x600")
            
            # Create analysis display
            self.analysis_text = tk.Text(
                self.analysis_window,
                wrap=tk.WORD,
                font=("Arial", 10),
                padx=10,
                pady=10
            )
            self.analysis_text.pack(fill='both', expand=True)
            
        # Update analysis
        analysis = self.analyzer.analyze_position(self.board, self.current_player)
        
        # Format analysis text
        text = "Current Position Analysis\n"
        text += "=" * 30 + "\n\n"
        
        # Board Control
        text += "Board Control:\n"
        control = analysis['control']
        text += f"Corners: Black {control['corners']['B']}-{control['corners']['W']} White\n"
        text += f"Edges: Black {control['edges']['B']}-{control['edges']['W']} White\n"
        text += f"Center: Black {control['center']['B']}-{control['center']['W']} White\n\n"
        
        # Mobility
        mobility = analysis['mobility']
        text += "Mobility:\n"
        text += f"Black: {mobility['B']} possible moves\n"
        text += f"White: {mobility['W']} possible moves\n"
        text += f"Mobility Ratio: {mobility['ratio']:.2f}\n\n"
        
        # Stability
        stability = analysis['stability']
        text += "Stable Pieces:\n"
        text += f"Black: {stability['B']}\n"
        text += f"White: {stability['W']}\n\n"
        
        # Potential
        potential = analysis['potential']
        text += "Territory Control:\n"
        text += f"Black: {potential['territory']['B']} potential moves\n"
        text += f"White: {potential['territory']['W']} potential moves\n"
        text += f"Black Frontier: {potential['frontier']['B']} pieces\n"
        text += f"White Frontier: {potential['frontier']['W']} pieces\n"
        
        # Update text widget
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', text)

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello - Main Menu")
        self.root.configure(bg='#F0F0F0')
        
        # Initialize sound manager
        self.sound_manager = SoundManager()
        
        # Load settings
        self.settings = self.load_settings()
        
        # Center the window
        window_width = 500
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create main menu widgets
        self.create_widgets()

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                return json.load(f)
        except:
            return {
                'move_time': 30,
                'difficulty': 'medium',
                'sound_volume': 50,
                'sound_theme': 'default',
                'reset_timer': True
            }

    def save_settings(self):
        try:
            with open('settings.json', 'w') as f:
                json.dump(self.settings, f)
        except:
            pass

    def save_seconds(self, event=None):
        try:
            value = int(self.seconds_entry.get())
            if 10 <= value <= 999:
                self.settings['move_time'] = value
                self.save_settings()
                self.sound_manager.play('button')
        except ValueError:
            # Reset to previous valid value
            self.seconds_entry.delete(0, tk.END)
            self.seconds_entry.insert(0, str(self.settings['move_time']))

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#F0F0F0')
        main_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        # Title with shadow effect
        title_frame = tk.Frame(main_frame, bg='#F0F0F0')
        title_frame.pack(pady=20)
        
        # OTHELLO text in green
        title_label = tk.Label(
            title_frame,
            text="OTHELLO",
            font=("Arial", 48, "bold"),
            bg='#F0F0F0',
            fg='#1B5E20'
        )
        title_label.pack()
        
        # BETA text
        beta_label = tk.Label(
            title_frame,
            text="BETA",
            font=("Arial", 16, "bold"),
            bg='#F0F0F0',
            fg='#4CAF50'
        )
        beta_label.pack(pady=(0, 20))

        # Time settings frame
        time_frame = tk.LabelFrame(
            main_frame,
            text="Time Settings",
            font=("Arial", 12, "bold"),
            bg='#F0F0F0'
        )
        time_frame.pack(pady=20, fill='x', padx=20)

        # Create a frame for time settings
        time_input_frame = tk.Frame(time_frame, bg='#F0F0F0')
        time_input_frame.pack(fill='x', padx=10, pady=5)

        # Time label
        time_label = tk.Label(
            time_input_frame,
            text="Seconds per move:",
            font=("Arial", 10),
            bg='#F0F0F0'
        )
        time_label.pack(side='left', padx=(0, 10))

        # Time entry field - using basic tk.Entry without validation
        self.seconds_entry = tk.Entry(
            time_input_frame,
            width=5,
            justify='center',
            font=("Arial", 12)
        )
        self.seconds_entry.insert(0, str(self.settings['move_time']))
        self.seconds_entry.pack(side='left')
        
        # Bind entry events
        self.seconds_entry.bind('<Return>', self.save_seconds)
        self.seconds_entry.bind('<FocusOut>', self.save_seconds)

        # Add "seconds" label
        seconds_label = tk.Label(
            time_input_frame,
            text="seconds",
            font=("Arial", 10),
            bg='#F0F0F0'
        )
        seconds_label.pack(side='left', padx=(5, 0))

        # Add hint label
        hint_label = tk.Label(
            time_frame,
            text="Enter a number between 10 and 999 seconds",
            font=("Arial", 8),
            fg='#666666',
            bg='#F0F0F0'
        )
        hint_label.pack(pady=(0, 5))

        # Timer reset option
        self.reset_timer_var = tk.BooleanVar(value=self.settings.get('reset_timer', True))
        reset_timer_check = ttk.Checkbutton(
            time_frame,
            text="Reset timer after each move",
            variable=self.reset_timer_var,
            command=self.update_reset_timer,
            style='Switch.TCheckbutton'
        )
        reset_timer_check.pack(pady=5)

        # Add description
        reset_timer_desc = tk.Label(
            time_frame,
            text="If disabled, the timer will continue counting down until the game ends",
            font=("Arial", 8),
            fg='#666666',
            bg='#F0F0F0',
            wraplength=400
        )
        reset_timer_desc.pack(pady=(0,5))
        
        # Statistics
        stats_frame = tk.LabelFrame(
            main_frame,
            text="Statistics",
            font=("Arial", 12, "bold"),
            bg='#F0F0F0'
        )
        stats_frame.pack(pady=20, fill='x', padx=20)
        
        # Initialize default stats structure
        default_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'best_time': float('inf'),
            'total_moves': 0,
            'difficulty_stats': {
                'easy': {'games_played': 0, 'wins': 0, 'losses': 0, 'best_time': float('inf'), 'total_moves': 0, 'average_score': 0, 'win_streak': 0, 'best_win_streak': 0},
                'medium': {'games_played': 0, 'wins': 0, 'losses': 0, 'best_time': float('inf'), 'total_moves': 0, 'average_score': 0, 'win_streak': 0, 'best_win_streak': 0},
                'hard': {'games_played': 0, 'wins': 0, 'losses': 0, 'best_time': float('inf'), 'total_moves': 0, 'average_score': 0, 'win_streak': 0, 'best_win_streak': 0},
                'impossible': {'games_played': 0, 'wins': 0, 'losses': 0, 'best_time': float('inf'), 'total_moves': 0, 'average_score': 0, 'win_streak': 0, 'best_win_streak': 0}
            }
        }
        
        try:
            with open('othello_stats.json', 'r') as f:
                stats = json.load(f)
                for key in default_stats:
                    if key not in stats:
                        stats[key] = default_stats[key]
                for diff in default_stats['difficulty_stats']:
                    if diff not in stats['difficulty_stats']:
                        stats['difficulty_stats'][diff] = default_stats['difficulty_stats'][diff]
        except:
            stats = default_stats
        
        stats_text = f"Total Games: {stats['games_played']}\n"
        stats_text += f"Total Wins: {stats['wins']}\n"
        stats_text += f"Total Losses: {stats['losses']}\n"
        if stats['best_time'] != float('inf'):
            stats_text += f"Best Time: {stats['best_time']:.1f}s\n"
        stats_text += f"Total Moves: {stats['total_moves']}"
        
        stats_label = tk.Label(
            stats_frame,
            text=stats_text,
            font=("Arial", 12),
            bg='#F0F0F0',
            justify='left'
        )
        stats_label.pack(pady=10, padx=10)
        
        # Difficulty selection
        difficulty_frame = tk.LabelFrame(
            main_frame,
            text="Select Difficulty",
            font=("Arial", 14, "bold"),
            bg='#F0F0F0'
        )
        difficulty_frame.pack(pady=20, fill='x', padx=20)
        
        self.difficulty_var = tk.StringVar(value=self.settings['difficulty'])
        
        difficulties = [
            ("Easy - Novice", "easy"),
            ("Medium - Strategist", "medium"),
            ("Hard - Master", "hard"),
            ("Impossible - Grandmaster", "impossible")
        ]
        
        for text, value in difficulties:
            frame = tk.Frame(difficulty_frame, bg='#F0F0F0')
            frame.pack(fill='x', pady=5, padx=10)
            
            radio = tk.Radiobutton(
                frame,
                text=text,
                variable=self.difficulty_var,
                value=value,
                font=("Arial", 12),
                bg='#F0F0F0',
                selectcolor='#4CAF50',
                activebackground='#F0F0F0',
                command=lambda: self.sound_manager.play('button')
            )
            radio.pack(side='left')
            
            # Add difficulty description
            if value in stats['difficulty_stats']:
                diff_stats = stats['difficulty_stats'][value]
                if diff_stats['games_played'] > 0:
                    win_rate = (diff_stats['wins'] / diff_stats['games_played']) * 100
                    label = tk.Label(
                        frame,
                        text=f"({win_rate:.1f}% win rate)",
                        font=("Arial", 10),
                        bg='#F0F0F0',
                        fg='#666666'
                    )
                    label.pack(side='right')
        
        # Start button with hover effect
        start_button = tk.Button(
            main_frame,
            text="Start Game",
            font=("Arial", 14, "bold"),
            command=self.start_game,
            bg='#4CAF50',
            fg='white',
            padx=30,
            pady=10,
            relief=tk.RAISED,
            borderwidth=2
        )
        start_button.pack(pady=30)
        
        # Bind hover events
        start_button.bind('<Enter>', lambda e: self.on_button_hover(start_button, True))
        start_button.bind('<Leave>', lambda e: self.on_button_hover(start_button, False))

    def on_button_hover(self, button, entering):
        if entering:
            button.config(bg='#45a049')
            self.sound_manager.play('button')
        else:
            button.config(bg='#4CAF50')

    def start_game(self):
        # Save current settings
        self.settings['difficulty'] = self.difficulty_var.get()
        self.settings['reset_timer'] = self.reset_timer_var.get()
        self.save_settings()
        
        # Clear main menu
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Start the game with selected settings
        game = Othello(
            self.root, 
            self.difficulty_var.get(), 
            self.settings['move_time'],
            self.reset_timer_var.get()
        )

    def update_reset_timer(self):
        self.settings['reset_timer'] = self.reset_timer_var.get()
        self.save_settings()
        self.sound_manager.play('button')

def main():
    root = tk.Tk()
    root.configure(bg='#F0F0F0')
    MainMenu(root)
    root.mainloop()

if __name__ == "__main__":
    main()
