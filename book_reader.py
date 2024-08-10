# load packages
from collections import OrderedDict
from pydub import AudioSegment

import re
import soundfile as sf
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from ML_model.models import build_model, load_ASR_models, load_F0_models
from ML_model.utils import recursive_munch
from ML_model.text_utils import TextCleaner

import torch
import random

import numpy as np

import yaml
from ML_model.Modules.diffusion.sampler import (
    DiffusionSampler,
    ADPM2Sampler,
    KarrasSchedule,
)
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from ML_model.Utils.PLBERT.util import load_plbert

import logging

import pymupdf


class BookReader:

    _MODEL_CFG_PATH = "book_reader_cfg.yml"

    def __init__(self, reference_audio_path=None, **kwargs):
        if not kwargs:
            kwargs = self.get_kwargs_from_yaml()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.reference_audio_path = reference_audio_path

        self.textcleaner = TextCleaner()

        self.preprocess_mean = -4
        self.preprocess_std = 4

        self._ESPEAK_LIBRARY_PATH = (
            "/opt/homebrew/Cellar/espeak-ng/1.51/lib/libespeak-ng.dylib"
        )

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.s_prev = None

        self.audio_samples = []

        return None

    @classmethod
    def get_kwargs_from_yaml(cls):
        yaml_file = cls._MODEL_CFG_PATH

        with open(yaml_file, "r") as file:
            kwargs = yaml.safe_load(file)

        return kwargs

    def set_device(self):

        print("Setting device.")
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Device set to {self.device}.")

        return None

    def set_path_to_reference_audio(self):
        if self.reference_audio_path is None:
            self.reference_audio_path = self.default_reference_audio_path
            print("Reference audio was set to default.")
        else:
            print(f"Reference audio path was taken from incoming file.")

        return None

    def preprocess(self, wave):

        wave_tensor = torch.from_numpy(wave).float()

        # generate spectrogram
        generate_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            n_mels=self.preprocess_params["spectrogram"]["n_mels"],
            n_fft=self.preprocess_params["spectrogram"]["n_fft"],
            win_length=self.preprocess_params["spectrogram"]["win_length"],
            hop_length=self.preprocess_params["spectrogram"]["hop_length"],
        )

        mel_tensor = generate_mel_spectrogram(wave_tensor)
        mel_tensor = (
            torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.preprocess_params["mean"]
        ) / self.preprocess_params["std"]

        return mel_tensor

    def compute_style(self):

        wave, sr = librosa.load(self.reference_audio_path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

            self.reference_audio_style = torch.cat([ref_s, ref_p], dim=1)
            return self.reference_audio_style

    def load_phonemizer(self):

        print("Loading phonemizer.")
        try:
            self.global_phonemizer = EspeakBackend(**self.phonemizer_settings)
        except Exception:
            EspeakWrapper.set_library(self._ESPEAK_LIBRARY_PATH)
            self.global_phonemizer = EspeakBackend(**self.phonemizer_settings)
        print("Phonemizer has been loaded.")

        return None

    def load_pretrained_models(self):

        print("Loading pretrained models - reading yml file.")
        self.pretrained_model_config = yaml.safe_load(
            open(self.pretrained_model_config_path)
        )

        print("Loading pretrained models - ASR model.")
        ASR_config = self.pretrained_model_config.get("ASR_config", False)
        ASR_path = self.pretrained_model_config.get("ASR_path", False)
        self.text_aligner = load_ASR_models(ASR_path, ASR_config)

        print("Loading pretrained models - F0 model.")
        F0_path = self.pretrained_model_config.get("F0_path", False)
        self.pitch_extractor = load_F0_models(F0_path)

        print("Loading pretrained models - BERT model.")
        BERT_path = self.pretrained_model_config.get("PLBERT_dir", False)
        self.plbert = load_plbert(BERT_path)

        print("Loading pretrained models - completed.")

    def create_model(self):
        self.model_params = recursive_munch(
            self.pretrained_model_config["model_params"]
        )
        self.model = build_model(
            self.model_params, self.text_aligner, self.pitch_extractor, self.plbert
        )
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load(self.whole_params_path, map_location="cpu")
        params = params_whole["net"]

        for key in self.model:

            if key in params:
                print("%s loaded" % key)

                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)

        _ = [self.model[key].eval() for key in self.model]

        return None

    def prepare_diffusion_sampler(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

        return None

    @staticmethod
    def length_to_mask(lengths):

        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))

        return mask

    def inference(self, text, output_file_path):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        ps = ps.replace("``", '"')
        ps = ps.replace("''", '"')

        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=self.embedding_scale,
                features=self.reference_audio_style,  # reference from the same speaker as the embedding
                num_steps=self.diffusion_steps,
            ).squeeze(1)

            if self.s_prev is not None:
                # convex combination of previous and current style
                s_pred = self.t * self.s_prev + (1 - self.t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = (
                self.alpha * ref
                + (1 - self.alpha) * self.reference_audio_style[:, :128]
            )
            s = self.beta * s + (1 - self.beta) * self.reference_audio_style[:, 128:]

            s_pred = torch.cat([ref, s], dim=-1)

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

            self.s_prev = s_pred

            wav = (
                out.squeeze().cpu().numpy()[..., :-100]
            )  # weird pulse at the end of the model, need to be fixed later
            sf.write(f"{output_file_path}", wav, 24000, "PCM_16")

        return (
            wav,
            s_pred,
        )  # weird pulse at the end of the model, need to be fixed later

    @staticmethod
    def split_paragraph(text, max_words=50):

        sentences = re.split(r"[.,;!?]", text)
        parts = [""]
        word_counter = 0

        for sentence in sentences:
            word_counter += len(sentence.split())
            if word_counter <= max_words:
                parts[-1] = " ".join([parts[-1], sentence])
            else:
                if len(sentence.split()) <= max_words:
                    parts.append(sentence)
                else:
                    while len(sentence.split()) > max_words:
                        parts.append(" ".join(sentence.split()[:max_words]))
                        sentence = " ".join(sentence.split()[max_words:])
        return parts

    def extract_paragraphs_from_book(self):
        doc = pymupdf.open(self.book_path)
        all_paragraphs = []

        for page_num in range(len(doc)):
            # for page_num in range(61, 62):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            print("BLOCKS")
            print(blocks)

            for block in blocks:
                if "lines" in block:
                    paragraph = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # span = self.clean_text(span)
                            if span["text"][-1] == "-":
                                paragraph += span["text"][:-1]
                            else:
                                paragraph += span["text"] + " "

                    paragraph = " ".join(paragraph.split())
                    if paragraph:
                        all_paragraphs.append(paragraph)

        self.processed_paragraphs = []
        for paragraph in all_paragraphs:
            if len(paragraph.split()) > 50:
                parts = self.split_paragraph(paragraph)
                print(f"parts are {parts}")
                self.processed_paragraphs.extend(parts)
            else:
                print(f"paragraph is small")
                self.processed_paragraphs.append(paragraph)

        print(self.processed_paragraphs)

        return None

    def merge_audio_output(self):
        combined = None
        pause_duration = 500
        pause = AudioSegment.silent(duration=pause_duration)

        for sample in self.audio_samples:
            sound = AudioSegment.from_wav(sample)
            if combined is None:
                combined = sound
            else:
                combined += sound + pause

        combined.export("audiobook.wav", format="wav")

        return None

    def read_book(self):
        self.set_device()
        self.load_phonemizer()
        self.load_pretrained_models()
        self.set_path_to_reference_audio()
        self.create_model()
        self.prepare_diffusion_sampler()
        self.compute_style()
        self.extract_paragraphs_from_book()

        for idx, paragraph in enumerate(self.processed_paragraphs, start=1):
            if paragraph:
                print(f"PARAGRAPH {idx}, len = {len(paragraph.split())}:")
                print(paragraph)

                self.inference(text=paragraph, output_file_path=f"{idx}.wav")
                self.audio_samples.extend([f"{idx}.wav"])
        self.merge_audio_output()

        return "audiobook.wav"


# import re

# text = "While they were doing this they discovered a lot of new and wonderful things that the pirates must have stolen from other ships: Kashmir shawls as thin as a cobweb, embroidered with flowers of gold; jars of fine tobacco from Jamaica; carved ivory boxes full of Russian tea; an old violin with a string broken and a picture on the back; a set of big chessmen, carved out of coral and amber; a walking-stick which had a sword inside it when you pulled the handle; six wine-glasses with turquoise and silver round the rims; and a lovely great sugar-bowl, made of mother o' pearl"

# re.split(r"[.;!?]", text)
