"""
Sistem Rekomendasi Game berbasis Content-Based Filtering
========================================================
Tugas Besar Kelompok AI Concepts (tema: Simple Recommendation)

Cara kerja singkat
------------------
1. Setiap game direpresentasikan oleh tiga fitur konten: genre, tag, dan deskripsi.
2. Tiap fitur diubah menjadi vektor TF-IDF (genre dan tag diperlakukan sebagai
   token utuh yang dipisah tanda "|", deskripsi diperlakukan sebagai teks biasa).
3. Ketiga vektor digabung dengan bobot (w_genre, w_tag, w_deskripsi). Karena tiap
   blok sudah ternormalisasi L2 dan diskalakan sqrt(bobot), hasil kali dalam dua
   vektor gabungan sama dengan jumlah berbobot dari cosine similarity tiap fitur:

       skor(a, b) = w_genre * cos(genre_a, genre_b)
                  + w_tag   * cos(tag_a, tag_b)
                  + w_desk  * cos(deskripsi_a, deskripsi_b)

4. Rekomendasi = game dengan skor tertinggi (kecuali game acuan itu sendiri).
   - Mode 1: acuan satu game.
   - Mode 2: acuan beberapa game favorit; profil pengguna = rata-rata vektor game
     yang disukai.

Contoh pemakaian
----------------
    python recommender.py                                   # mode interaktif
    python recommender.py --judul "Hollow Knight" -k 5
    python recommender.py --suka Hades Celeste "Dead Cells" -k 5
    python recommender.py --judul Minecraft --genre Survival
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_DEFAULT = Path(__file__).with_name("games.csv")
BOBOT_DEFAULT = {"genre": 0.25, "tag": 0.50, "deskripsi": 0.25}


class JudulTidakDitemukan(Exception):
    """Dilempar saat judul tidak ada di dataset atau cocok dengan banyak game."""

    def __init__(self, query, saran=None, ambigu=False):
        self.query = query
        self.saran = list(saran or [])
        self.ambigu = ambigu
        pesan = (
            f"Judul '{query}' cocok dengan lebih dari satu game."
            if ambigu
            else f"Judul '{query}' tidak ditemukan."
        )
        super().__init__(pesan)


def _pecah(teks):
    """Memecah string 'A|B|C' menjadi token huruf kecil ['a', 'b', 'c']."""
    return [t.strip().lower() for t in str(teks).split("|") if t.strip()]


class ContentBasedRecommender:
    """Rekomender game berbasis konten (TF-IDF + cosine similarity berbobot)."""

    def __init__(self, csv_path=CSV_DEFAULT, bobot=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.df["title_lower"] = self.df["title"].str.lower()
        self.df["genre_set"] = self.df["genres"].map(lambda s: set(_pecah(s)))
        self.df["tag_set"] = self.df["tags"].map(lambda s: set(_pecah(s)))
        self.bobot = self._normalisasi_bobot(bobot or BOBOT_DEFAULT)
        self._bangun_matriks()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _normalisasi_bobot(bobot):
        w = {k: float(bobot.get(k, 0.0)) for k in ("genre", "tag", "deskripsi")}
        if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError("Bobot harus >= 0 dan minimal satu bobot > 0.")
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def _bangun_matriks(self):
        """Membangun matriks fitur gabungan X (jumlah_game x jumlah_fitur)."""
        blok = []
        if self.bobot["genre"] > 0:
            vec = TfidfVectorizer(tokenizer=_pecah, lowercase=False, token_pattern=None)
            blok.append(vec.fit_transform(self.df["genres"]) * float(np.sqrt(self.bobot["genre"])))
        if self.bobot["tag"] > 0:
            vec = TfidfVectorizer(tokenizer=_pecah, lowercase=False, token_pattern=None)
            blok.append(vec.fit_transform(self.df["tags"]) * float(np.sqrt(self.bobot["tag"])))
        if self.bobot["deskripsi"] > 0:
            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
            blok.append(vec.fit_transform(self.df["description"]) * float(np.sqrt(self.bobot["deskripsi"])))
        self.X = hstack(blok, format="csr")
        self._norma = np.asarray(np.sqrt(self.X.multiply(self.X).sum(axis=1))).ravel()

    # ---------------------------------------------------------------- pencarian
    def daftar_genre(self):
        semua = set()
        for s in self.df["genre_set"]:
            semua |= s
        return sorted(semua)

    def cari_judul(self, query):
        """Mengembalikan indeks baris untuk judul. Toleran huruf besar/kecil dan judul parsial."""
        q = (query or "").strip().lower()
        if not q:
            raise JudulTidakDitemukan(query)
        judul = self.df["title_lower"]
        persis = np.flatnonzero(judul.to_numpy() == q)
        if len(persis):
            return int(persis[0])
        sebagian = np.flatnonzero(judul.str.contains(re.escape(q)).to_numpy())
        if len(sebagian) == 1:
            return int(sebagian[0])
        if len(sebagian) > 1:
            raise JudulTidakDitemukan(query, self.df.loc[sebagian[:5], "title"].tolist(), ambigu=True)
        mirip = difflib.get_close_matches(q, judul.tolist(), n=3, cutoff=0.5)
        saran = [self.df.loc[judul == m, "title"].iloc[0] for m in mirip]
        raise JudulTidakDitemukan(query, saran)

    # ------------------------------------------------------------------- skor
    def _cosine_ke_semua(self, vektor):
        skor = self.X @ vektor
        penyebut = self._norma * np.linalg.norm(vektor)
        return np.divide(skor, penyebut, out=np.zeros_like(skor, dtype=float), where=penyebut > 0)

    def skor_game(self, idx):
        """Skor kemiripan satu game (indeks idx) terhadap semua game."""
        return self._cosine_ke_semua(self.X[idx].toarray().ravel())

    def skor_profil(self, daftar_idx):
        """Skor kemiripan profil (rata-rata vektor game favorit) terhadap semua game."""
        profil = np.asarray(self.X[list(daftar_idx)].mean(axis=0)).ravel()
        return self._cosine_ke_semua(profil)

    # ------------------------------------------------------------ rekomendasi
    def _alasan(self, i, acuan):
        hitung = Counter()
        for a in acuan:
            hitung.update(self.df.at[a, "genre_set"] | self.df.at[a, "tag_set"])
        bersama = (self.df.at[i, "genre_set"] | self.df.at[i, "tag_set"]) & set(hitung)
        urut = sorted(bersama, key=lambda t: (-hitung[t], t))
        return ", ".join(urut[:5])

    def _susun_hasil(self, skor, kecualikan, k, genre, acuan):
        if k <= 0:
            raise ValueError("Jumlah rekomendasi (k) harus lebih dari 0.")
        skor = skor.astype(float).copy()
        skor[list(kecualikan)] = -np.inf
        if genre:
            g = genre.strip().lower()
            if g not in {x.lower() for x in self.daftar_genre()}:
                raise ValueError(f"Genre '{genre}' tidak dikenal. Pilihan: {', '.join(self.daftar_genre())}")
            cocok = self.df["genre_set"].map(lambda s: g in s).to_numpy()
            skor[~cocok] = -np.inf
        urutan = [i for i in np.argsort(-skor, kind="stable")[:k] if np.isfinite(skor[i])]
        baris = [
            {
                "peringkat": r,
                "judul": self.df.at[i, "title"],
                "tahun": int(self.df.at[i, "year"]),
                "genre": self.df.at[i, "genres"].replace("|", " / "),
                "skor": round(float(skor[i]), 4),
                "alasan": self._alasan(i, acuan),
            }
            for r, i in enumerate(urutan, start=1)
        ]
        return pd.DataFrame(baris, columns=["peringkat", "judul", "tahun", "genre", "skor", "alasan"])

    def rekomendasi_game(self, judul, k=5, genre=None):
        """Rekomendasi game yang mirip dengan satu game acuan."""
        idx = self.cari_judul(judul)
        return self._susun_hasil(self.skor_game(idx), [idx], k, genre, [idx])

    def rekomendasi_profil(self, daftar_judul, k=5, genre=None):
        """Rekomendasi berdasarkan beberapa game favorit (profil pengguna)."""
        if not daftar_judul:
            raise ValueError("Daftar game favorit tidak boleh kosong.")
        idx = sorted({self.cari_judul(j) for j in daftar_judul})
        return self._susun_hasil(self.skor_profil(idx), idx, k, genre, idx)


# ------------------------------------------------------------------ antarmuka
def skor_ke_persen(skor):
    """0.545 -> '54,5%' (format desimal koma, sama seperti evaluasi.py)."""
    return f"{skor * 100:.1f}%".replace(".", ",")


def skor_ke_rating(skor):
    """Mengubah skor kemiripan 0-1 menjadi rating bulat 1-5 (linear: skor x 5, minimal 1)."""
    return int(min(5, max(1, round(skor * 5))))


def _bintang(rating):
    """Rating 1-5 -> '★★★☆☆'. Memakai '*' dan '-' bila terminal tidak mendukung Unicode."""
    penuh, kosong = "★", "☆"
    try:
        (penuh + kosong).encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        penuh, kosong = "*", "-"
    return penuh * rating + kosong * (5 - rating)


def cetak_hasil(df):
    if df.empty:
        print("Tidak ada rekomendasi yang cocok.")
        return
    for _, r in df.iterrows():
        rating = skor_ke_rating(r.skor)
        print(f"{r.peringkat}. {r.judul} ({r.tahun})")
        print(f"   Kecocokan: {skor_ke_persen(r.skor)}   Rating: {_bintang(rating)} ({rating}/5)")
        print(f"   Genre    : {r.genre}")
        print(f"   Alasan   : {r.alasan or '-'}")


def _tampilkan_error_judul(e):
    print(f"! {e}")
    if e.saran:
        print("  Maksud Anda: " + "; ".join(e.saran))


def mode_interaktif(rek):
    print("=" * 60)
    print(" SISTEM REKOMENDASI GAME (Content-Based Filtering)")
    print(f" Dataset: {len(rek.df)} game")
    print("=" * 60)
    while True:
        print("\nMENU")
        print(" 1. Rekomendasi dari satu game")
        print(" 2. Rekomendasi dari beberapa game favorit")
        print(" 3. Lihat daftar game")
        print(" 4. Keluar")
        try:
            pilih = input("Pilih menu (1-4): ").strip()
            if pilih == "4":
                print("Terima kasih.")
                return
            if pilih == "3":
                for _, r in rek.df.iterrows():
                    print(f" - {r.title} ({r.year})  [{r.genres.replace('|', ' / ')}]")
                continue
            if pilih not in ("1", "2"):
                print("! Pilihan tidak valid.")
                continue
            if pilih == "1":
                masukan = [input("Judul game yang Anda sukai: ").strip()]
            else:
                masukan = [j.strip() for j in input("Judul game favorit (pisahkan dengan koma): ").split(",") if j.strip()]
            k_txt = input("Jumlah rekomendasi [5]: ").strip()
            k = int(k_txt) if k_txt else 5
            genre = input("Filter genre (kosongkan jika tidak perlu): ").strip() or None
            print()
            hasil = rek.rekomendasi_game(masukan[0], k, genre) if pilih == "1" else rek.rekomendasi_profil(masukan, k, genre)
            cetak_hasil(hasil)
        except JudulTidakDitemukan as e:
            _tampilkan_error_judul(e)
        except ValueError as e:
            print(f"! {e}")
        except (EOFError, KeyboardInterrupt):
            print("\nTerima kasih.")
            return


def main(argv=None):
    p = argparse.ArgumentParser(description="Rekomendasi game berbasis Content-Based Filtering")
    p.add_argument("--judul", help="Satu game acuan")
    p.add_argument("--suka", nargs="+", help="Beberapa game favorit")
    p.add_argument("-k", type=int, default=5, help="Jumlah rekomendasi (default 5)")
    p.add_argument("--genre", help="Filter genre hasil rekomendasi")
    p.add_argument("--csv", default=str(CSV_DEFAULT), help="Lokasi dataset CSV")
    a = p.parse_args(argv)

    rek = ContentBasedRecommender(a.csv)
    if not a.judul and not a.suka:
        mode_interaktif(rek)
        return 0
    try:
        if a.judul:
            print(f"Game yang mirip dengan '{a.judul}':\n")
            cetak_hasil(rek.rekomendasi_game(a.judul, a.k, a.genre))
        else:
            print(f"Rekomendasi berdasarkan favorit: {', '.join(a.suka)}\n")
            cetak_hasil(rek.rekomendasi_profil(a.suka, a.k, a.genre))
    except JudulTidakDitemukan as e:
        _tampilkan_error_judul(e)
        return 1
    except ValueError as e:
        print(f"! {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
