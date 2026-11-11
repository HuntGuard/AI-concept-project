"""
Antarmuka web (Streamlit) untuk Sistem Rekomendasi Game.
Mesin rekomendasinya tetap memakai recommender.py, jadi hasilnya sama dengan versi CLI.

Jalankan lokal:  streamlit run app.py
"""
import streamlit as st

from recommender import ContentBasedRecommender, skor_ke_persen, skor_ke_rating

st.set_page_config(page_title="Rekomendasi Game", page_icon="🎮", layout="centered")


@st.cache_resource
def muat_rekomender():
    """Dataset dan matriks TF-IDF dibangun sekali saja, lalu dipakai ulang."""
    return ContentBasedRecommender()


def bintang(rating):
    return "★" * rating + "☆" * (5 - rating)


rek = muat_rekomender()
daftar_judul = sorted(rek.df["title"].tolist())

st.title("🎮 Sistem Rekomendasi Game")
st.caption(f"Content-Based Filtering (TF-IDF + cosine similarity) · {len(rek.df)} game")

mode = st.radio(
    "Mode rekomendasi",
    ["Dari satu game", "Dari beberapa game favorit"],
    horizontal=True,
)

if mode == "Dari satu game":
    pilihan = st.selectbox("Game yang Anda sukai", daftar_judul)
else:
    pilihan = st.multiselect("Game favorit Anda", daftar_judul)

kol1, kol2 = st.columns(2)
with kol1:
    k = st.slider("Jumlah rekomendasi", min_value=1, max_value=20, value=5)
with kol2:
    genre = st.selectbox(
        "Filter genre",
        [None] + rek.daftar_genre(),
        format_func=lambda g: "Semua genre" if g is None else g.title(),
    )

if st.button("Cari rekomendasi", type="primary"):
    try:
        if mode == "Dari satu game":
            hasil = rek.rekomendasi_game(pilihan, k=k, genre=genre)
        else:
            hasil = rek.rekomendasi_profil(pilihan, k=k, genre=genre)
    except Exception as e:  # judul tidak ditemukan, daftar kosong, genre/k tidak valid
        st.warning(str(e))
    else:
        if hasil.empty:
            st.info("Tidak ada rekomendasi yang cocok.")
        for _, r in hasil.iterrows():
            rating = skor_ke_rating(r.skor)
            with st.container(border=True):
                st.markdown(f"**{r.peringkat}. {r.judul}** ({r.tahun})")
                st.caption(r.genre)
                st.progress(float(min(max(r.skor, 0.0), 1.0)))
                st.write(f"Kecocokan **{skor_ke_persen(r.skor)}** · Rating {bintang(rating)} ({rating}/5)")
                st.write(f"Alasan: {r.alasan or '-'}")
