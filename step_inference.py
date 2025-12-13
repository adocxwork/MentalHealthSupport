#!/usr/bin/env python3
"""
Improved inference script — maps generic label_# names to real emotion names.
Replace your existing improved_inference.py with this file.
"""
import torch
import json
import re
import os
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from collections import defaultdict
from typing import List, Optional, Dict, Any
from tqdm import tqdm

# Default fallback emotion list (GoEmotions-like 28 labels — matches the mapping you showed)
GOEMOTIONS_DEFAULT = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

class ImprovedEmotionDetector:
    def __init__(self,
                 model_path: str = 'emotion_roberta_model',
                 device: Optional[str] = None,
                 max_length: int = 256,
                 batch_size: int = 16):
        print("🤖 Loading emotion detection model...")
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"   Device: {self.device}")

        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

        # Load mapping and thresholds
        self.emotions = self._load_emotion_mapping(model_path)
        print(f"   Loaded {len(self.emotions)} emotion classes")
        self.thresholds = self._load_thresholds(model_path)
        print("✅ Model loaded successfully!\n")

    def _load_emotion_mapping(self, model_path: str) -> List[str]:
        """Robust emotion mapping. Falls back to GOEMOTIONS_DEFAULT when labels are generic."""
        def _dict_to_list(idx_map: Dict[Any, str]) -> List[str]:
            try:
                sorted_items = sorted(((int(k), v) for k, v in idx_map.items()), key=lambda x: x[0])
                return [v for _, v in sorted_items]
            except Exception:
                return list(idx_map.values())

        # 1) emotion_mapping.json (check model directory first, then root)
        mapping_paths = [
            os.path.join(model_path, 'emotion_mapping.json'),
            'emotion_mapping.json'  # Also check root directory
        ]
        for mapping_path in mapping_paths:
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                    if isinstance(mapping, dict):
                        if 'emotions' in mapping and isinstance(mapping['emotions'], list):
                            return [str(e) for e in mapping['emotions']]
                        if 'idx_to_emotion' in mapping and isinstance(mapping['idx_to_emotion'], dict):
                            return [str(e) for e in _dict_to_list(mapping['idx_to_emotion'])]
                        if 'emotion_to_idx' in mapping and isinstance(mapping['emotion_to_idx'], dict):
                            try:
                                inv = {int(v): k for k, v in mapping['emotion_to_idx'].items()}
                                return [inv[i] for i in range(max(inv.keys()) + 1)]
                            except Exception:
                                pass
                    elif isinstance(mapping, list):
                        return [str(e) for e in mapping]
                except Exception as e:
                    print(f"⚠️ Could not parse {mapping_path}: {e}")
                break  # If we found and tried to parse a file, don't check other paths

        # 2) best_config.json
        config_path = os.path.join(model_path, 'best_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    if 'emotions' in cfg and isinstance(cfg['emotions'], list):
                        return [str(e) for e in cfg['emotions']]
                    if 'idx_to_emotion' in cfg and isinstance(cfg['idx_to_emotion'], dict):
                        return [str(e) for e in _dict_to_list(cfg['idx_to_emotion'])]
                    if 'emotion_to_idx' in cfg and isinstance(cfg['emotion_to_idx'], dict):
                        try:
                            inv = {int(v): k for k, v in cfg['emotion_to_idx'].items()}
                            return [inv[i] for i in range(max(inv.keys()) + 1)]
                        except Exception:
                            pass
            except Exception as e:
                print(f"⚠️ Could not parse {config_path}: {e}")

        # 3) Try HuggingFace config id2label / label2id
        try:
            cfg = getattr(self.model, 'config', None)
            if cfg is not None:
                id2label = getattr(cfg, 'id2label', None)
                if id2label and isinstance(id2label, dict):
                    try:
                        id_label_list = [label for _, label in sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])]
                    except Exception:
                        id_label_list = list(id2label.values())

                    # If labels are like 'label_25' or 'LABEL_25', map using numbers to GOEMOTIONS_DEFAULT
                    generic_label_pattern = re.compile(r'^(?:label_|LABEL_)(\d+)$')
                    if all(generic_label_pattern.match(str(l)) for l in id_label_list):
                        mapped = []
                        for l in id_label_list:
                            m = generic_label_pattern.match(str(l))
                            idx = int(m.group(1))
                            if idx < len(GOEMOTIONS_DEFAULT):
                                mapped.append(GOEMOTIONS_DEFAULT[idx])
                            else:
                                mapped.append(f"emotion_{idx}")
                        print("   Note: model.config.id2label contained generic label_* names — using built-in emotion mapping.")
                        return mapped

                    # If not generic, return labels normalized
                    return [str(l).lower() for l in id_label_list]

                label2id = getattr(cfg, 'label2id', None)
                if label2id and isinstance(label2id, dict):
                    # Try to invert label2id
                    try:
                        inv = {int(v): k for k, v in label2id.items()}
                        # If keys are numeric indices and values are emotion strings, use them.
                        labels = [inv[i] for i in range(max(inv.keys()) + 1)]
                        # detect generic pattern
                        generic_label_pattern = re.compile(r'^(?:label_|LABEL_)(\d+)$')
                        if all(generic_label_pattern.match(str(l)) for l in labels):
                            mapped = []
                            for l in labels:
                                m = generic_label_pattern.match(str(l))
                                idx = int(m.group(1))
                                if idx < len(GOEMOTIONS_DEFAULT):
                                    mapped.append(GOEMOTIONS_DEFAULT[idx])
                                else:
                                    mapped.append(f"emotion_{idx}")
                            print("   Note: model.config.label2id contained generic label_* names — using built-in emotion mapping.")
                            return mapped
                        return [str(l).lower() for l in labels]
                    except Exception:
                        return [str(k).lower() for k in label2id.keys()]
        except Exception as e:
            print(f"⚠️ Error reading model.config for labels: {e}")

        # 4) Fallback: use built-in GOEMOTIONS_DEFAULT if model.num_labels matches or we prefer robust mapping
        n_labels = getattr(self.model.config, 'num_labels', None) or 0
        if n_labels and n_labels == len(GOEMOTIONS_DEFAULT):
            print("⚠️ No mapping files found — using built-in GOEMOTIONS_DEFAULT mapping.")
            return GOEMOTIONS_DEFAULT.copy()
        elif n_labels and n_labels > 0:
            # try to use default up to n_labels, otherwise generate emotion_i
            print("⚠️ No mapping files found — using built-in mapping for available indices where possible.")
            mapped = []
            for i in range(n_labels):
                if i < len(GOEMOTIONS_DEFAULT):
                    mapped.append(GOEMOTIONS_DEFAULT[i])
                else:
                    mapped.append(f"emotion_{i}")
            return mapped

        # ultimate fallback: empty
        print("⚠️ No label info available; returning empty label list.")
        return []

    def _load_thresholds(self, model_path: str) -> np.ndarray:
        config_path = os.path.join(model_path, 'best_config.json')
        default_threshold = 0.5
        n_emotions = len(self.emotions)

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if 'thresholds' in config and config['thresholds']:
                    thresholds = np.array(config['thresholds'], dtype=float)
                    if len(thresholds) != n_emotions:
                        print(f"⚠️ Threshold length ({len(thresholds)}) != number of emotions ({n_emotions}). Adjusting...")
                        if len(thresholds) < n_emotions:
                            pad = np.full(n_emotions - len(thresholds), default_threshold, dtype=float)
                            thresholds = np.concatenate([thresholds, pad])
                        else:
                            thresholds = thresholds[:n_emotions]
                    print(f"   Loaded optimized thresholds (range: {thresholds.min():.2f}-{thresholds.max():.2f})")
                    return thresholds
            except Exception as e:
                print(f"⚠️ Could not load thresholds from {config_path}: {e}")

        print(f"   Using default threshold: {default_threshold}")
        return np.full(n_emotions, default_threshold, dtype=float)

    def predict_single(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.predict_batch([text], top_k=top_k)[0]

    def predict_batch(self, texts: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        all_results = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in encoding.items() if k in ('input_ids', 'attention_mask')}
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))

                for prob_vector in probs:
                    emotions = []
                    for idx, prob in enumerate(prob_vector):
                        if idx >= len(self.emotions):
                            break
                        if prob >= self.thresholds[idx]:
                            emotions.append({
                                'emotion': self.emotions[idx],
                                'confidence': float(prob),
                                'idx': int(idx)
                            })

                    if not emotions:
                        top_indices = np.argsort(prob_vector)[-top_k:][::-1]
                        emotions = [
                            {
                                'emotion': self.emotions[idx] if idx < len(self.emotions) else f"emotion_{idx}",
                                'confidence': float(prob_vector[idx]),
                                'idx': int(idx)
                            }
                            for idx in top_indices
                        ]
                    else:
                        emotions = sorted(emotions, key=lambda x: x['confidence'], reverse=True)[:top_k]

                    all_results.append(emotions)
        return all_results

    def parse_chat(self, chat_text: str) -> List[tuple]:
        messages = []
        lines = chat_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            patterns = [
                r'^([^:]+):\s*(.+)$',
                r'^\[([^\]]+)\]\s*(.+)$',
                r'^([^-]+)-\s*(.+)$',
            ]
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    username = match.group(1).strip()
                    message = match.group(2).strip()
                    messages.append((username, message))
                    matched = True
                    break
            if not matched:
                if messages:
                    user, prev_msg = messages[-1]
                    messages[-1] = (user, prev_msg + " " + line)
                else:
                    messages.append(("Unknown", line))
        return messages

    def analyze_conversation(self, chat_text: str, top_k: int = 3) -> Dict[str, Any]:
        messages = self.parse_chat(chat_text)
        if not messages:
            return {'messages': [], 'per_user': {}, 'meta': {'n_messages': 0, 'n_users': 0, 'error': 'No messages found'}}
        usernames = [u for u, _ in messages]
        texts = [m for _, m in messages]
        print(f"\n📊 Analyzing conversation...")
        print(f"   Messages: {len(messages)}")
        print(f"   Users: {len(set(usernames))}")
        print("   Detecting emotions...")
        predictions = self.predict_batch(texts, top_k=top_k)
        per_message = []
        user_emotions = defaultdict(list)
        for (username, text), pred_list in zip(messages, predictions):
            per_message.append({'username': username, 'text': text, 'emotions': pred_list})
            if pred_list:
                user_emotions[username].append(pred_list[0]['emotion'])
        per_user = {}
        for username, emotion_list in user_emotions.items():
            emotion_counts = defaultdict(int)
            for emotion in emotion_list:
                emotion_counts[emotion] += 1
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            top_emotions = [e for e, _ in sorted_emotions[:top_k]]
            per_user[username] = {
                'top_emotions': top_emotions,
                'summary': ', '.join(top_emotions),
                'total_messages': len(emotion_list),
                'emotion_breakdown': dict(sorted_emotions)
            }
        return {'messages': per_message, 'per_user': per_user, 'meta': {'n_messages': len(messages), 'n_users': len(set(usernames))}}

    def pretty_print_single(self, text: str, predictions: List[Dict[str, Any]]):
        print(f"\n{'='*70}")
        print(f"📝 Text: {text}")
        print(f"{'='*70}")
        print("🎭 Detected Emotions:")
        for i, pred in enumerate(predictions, 1):
            conf = max(0.0, min(1.0, pred['confidence']))
            bar = '█' * int(conf * 30)
            print(f"   {i}. {pred['emotion']:15s} {conf:.3f} {bar}")
        print(f"{'='*70}\n")

    def pretty_print_conversation(self, results: Dict[str, Any]):
        print(f"\n{'='*70}")
        print("💬 CONVERSATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Total Messages: {results['meta']['n_messages']}")
        print(f"Total Users: {results['meta']['n_users']}")
        print(f"{'='*70}\n")
        print("👥 USER EMOTION SUMMARY")
        print(f"{'='*70}")
        for username, user_data in results['per_user'].items():
            print(f"\n{username} ({user_data['total_messages']} messages)")
            print(f"   Top Emotions: {user_data['summary']}")
            breakdown = user_data['emotion_breakdown']
            sorted_breakdown = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_breakdown[:5]:
                bar = '▓' * count
                print(f"   • {emotion:15s} {count:2d}x {bar}")
        print(f"\n{'='*70}\n")
        print("📨 MESSAGE DETAILS (first 10)")
        print(f"{'='*70}")
        for i, msg in enumerate(results['messages'][:10], 1):
            emotions_str = ', '.join([f"{e['emotion']}({e['confidence']:.2f})" for e in msg['emotions'][:3]])
            print(f"{i}. [{msg['username']}]: {msg['text'][:50]}...")
            print(f"   Emotions: {emotions_str}\n")
        if len(results['messages']) > 10:
            print(f"   ... and {len(results['messages']) - 10} more messages\n")
        print(f"{'='*70}\n")


def interactive_mode(detector: ImprovedEmotionDetector):
    print("="*70)
    print("🎯 INTERACTIVE EMOTION DETECTION MODE")
    print("="*70)
    print("Options:")
    print("  1. Analyze single text")
    print("  2. Analyze conversation/chat")
    print("  3. Batch analyze from file")
    print("  4. Exit")
    print("="*70)
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            if choice == '1':
                print("\n--- Single Text Analysis ---")
                text = input("Enter text: ").strip()
                if not text:
                    print("❌ Empty text!")
                    continue
                predictions = detector.predict_single(text, top_k=5)
                detector.pretty_print_single(text, predictions)
                save = input("Save results? (y/n): ").strip().lower()
                if save == 'y':
                    output_file = input("Output filename [results.json]: ").strip() or "results.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({'text': text, 'emotions': predictions}, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved to {output_file}")
            elif choice == '2':
                print("\n--- Conversation Analysis ---")
                print("Enter chat messages (format: 'Username: message')")
                print("Enter empty line or 'DONE' when finished:")
                lines = []
                while True:
                    line = input()
                    if not line or line.strip().upper() == 'DONE':
                        break
                    lines.append(line)
                if not lines:
                    print("❌ No messages entered!")
                    continue
                chat_text = '\n'.join(lines)
                results = detector.analyze_conversation(chat_text, top_k=3)
                detector.pretty_print_conversation(results)
                save = input("Save results? (y/n): ").strip().lower()
                if save == 'y':
                    output_file = input("Output filename [conversation_results.json]: ").strip() or "conversation_results.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved to {output_file}")
            elif choice == '3':
                print("\n--- Batch Analysis from File ---")
                input_file = input("Enter input filename: ").strip()
                if not os.path.exists(input_file):
                    print(f"❌ File not found: {input_file}")
                    continue
                mode = input("File type? (1=chat, 2=text list): ").strip()
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if mode == '1':
                    results = detector.analyze_conversation(content, top_k=3)
                    detector.pretty_print_conversation(results)
                    output_file = input("Save to [conversation_results.json]: ").strip() or "conversation_results.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved to {output_file}")
                else:
                    texts = [line.strip() for line in content.split('\n') if line.strip()]
                    print(f"\nAnalyzing {len(texts)} texts...")
                    predictions = detector.predict_batch(texts, top_k=5)
                    results = []
                    for text, preds in zip(texts, predictions):
                        results.append({'text': text, 'emotions': preds})
                    for i, result in enumerate(results[:5], 1):
                        print(f"\n{i}. {result['text'][:60]}...")
                        emotions = ', '.join([f"{e['emotion']}({e['confidence']:.2f})" for e in result['emotions'][:3]])
                        print(f"   {emotions}")
                    if len(results) > 5:
                        print(f"\n... and {len(results) - 5} more")
                    output_file = input("\nSave to [batch_results.json]: ").strip() or "batch_results.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved to {output_file}")
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option!")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emotion Detection Inference")
    parser.add_argument('--model_path', type=str, default='emotion_roberta_model', help='Path to trained model')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--chat_file', type=str, help='Chat file to analyze')
    parser.add_argument('--batch_file', type=str, help='File with texts to analyze (one per line)')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON file')
    parser.add_argument('--top_k', type=int, default=5, help='Top K emotions to return')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()

    detector = ImprovedEmotionDetector(model_path=args.model_path, max_length=256, batch_size=16)
    print("Sample loaded emotion labels:", detector.emotions[:10])

    if args.interactive or (not args.text and not args.chat_file and not args.batch_file):
        interactive_mode(detector)
        return

    if args.text:
        predictions = detector.predict_single(args.text, top_k=args.top_k)
        detector.pretty_print_single(args.text, predictions)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({'text': args.text, 'emotions': predictions}, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {args.output}")
    elif args.chat_file:
        with open(args.chat_file, 'r', encoding='utf-8') as f:
            chat_text = f.read()
        results = detector.analyze_conversation(chat_text, top_k=args.top_k)
        detector.pretty_print_conversation(results)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {args.output}")
    elif args.batch_file:
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"\n📊 Analyzing {len(texts)} texts...")
        predictions = detector.predict_batch(texts, top_k=args.top_k)
        results = []
        for text, preds in zip(texts, predictions):
            results.append({'text': text, 'emotions': preds})
        print("\n📝 Sample Results:")
        for i, result in enumerate(results[:5], 1):
            emotions = ', '.join([f"{e['emotion']}({e['confidence']:.2f})" for e in result['emotions'][:3]])
            print(f"{i}. {result['text'][:60]}...")
            print(f"   {emotions}\n")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
