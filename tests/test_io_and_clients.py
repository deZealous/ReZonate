import io
import zipfile

import numpy as np

from rezonate.embed import store
from rezonate.enrich.clients import LastFM, LRCLib, MusicBrainz
from rezonate.enrich.http import HTTPRangeFile
from rezonate.ingest.jamendo import Jamendo


class FakeHTTP:
    """Stands in for PoliteClient: returns canned JSON keyed by URL suffix."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get_json(self, url, params=None, **kw):
        self.calls.append((url, params))
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp(params) if callable(resp) else resp
        return None


def test_lrclib_get_then_search_fallback():
    http = FakeHTTP({"/api/get": None,
                     "/api/search": [{"id": 7, "duration": 201.0, "plainLyrics": "la la",
                                      "instrumental": False}]})
    r = LRCLib(http).get("A", "T", duration_s=200)
    assert r == {"plain": "la la", "instrumental": False, "lrclib_id": 7}
    assert http.calls[0][1]["duration"] == 200


def test_musicbrainz_takes_earliest_year_across_matching_recordings():
    resp = {"recordings": [
        {"id": "live", "score": 100, "title": "In the End", "first-release-date": "2007-05-01",
         "tags": [{"name": "rock", "count": 2}]},
        {"id": "orig", "score": 98, "title": "In The End", "first-release-date": "2000-10-24"},
        {"id": "other", "score": 95, "title": "In the End (Remix)", "first-release-date": "1990"},
    ]}
    r = MusicBrainz(FakeHTTP({"/recording": resp})).recording("Linkin Park", "In the End")
    assert r["year"] == 2000 and r["tags"] == [("rock", 2.0)]


def test_lastfm_tags_single_and_list():
    single = {"toptags": {"tag": {"name": "Post-Punk", "count": 100}}}
    out = LastFM("k", FakeHTTP({"/2.0/": single})).top_tags("a", "t")
    assert out == [("post-punk", 1.0)]


def test_lastfm_listeners_coerces_string():
    lfm = LastFM("k", FakeHTTP({"/2.0/": {"track": {"listeners": "12345"}}}))
    assert lfm.listeners("a", "t") == 12345.0
    assert LastFM("k", FakeHTTP({"/2.0/": {"error": 6}})).listeners("a", "t") is None


def test_jamendo_respects_download_flag():
    res = [{"id": "1", "name": "ok", "artist_name": "a", "audiodownload_allowed": True,
            "audiodownload": "http://x/1.mp3", "releasedate": "2015-03-02",
            "musicinfo": {"tags": {"genres": ["dub"], "instruments": [], "vartags": []}},
            "stats": {"listened": 10}},
           {"id": "2", "name": "no", "audiodownload_allowed": False}]
    df = Jamendo.to_frame(res)
    assert list(df.track_id) == ["jamendo:1"] and df.year.item() == 2015 and df.tags.item() == ["dub"]


def test_http_range_file_reads_zip_members():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("fma_large/000/000002.mp3", b"x" * 5000)
        zf.writestr("fma_large/000/000005.mp3", b"y" * 3000)
    data = buf.getvalue()

    class Resp:
        def __init__(self, content=b"", headers=None):
            self.content, self.headers = content, headers or {}

        def raise_for_status(self):
            pass

    class Sess:
        n = 0

        def head(self, url, **kw):
            return Resp(headers={"Content-Length": str(len(data))})

        def get(self, url, headers, **kw):
            Sess.n += 1
            a, b = map(int, headers["Range"].split("=")[1].split("-"))
            return Resp(data[a:b + 1])

    rf = HTTPRangeFile("http://x/a.zip", session=Sess(), block_size=1024)
    with zipfile.ZipFile(rf) as zf:
        assert zf.read("fma_large/000/000005.mp3") == b"y" * 3000
    assert Sess.n < len(data) // 1024 + 3


def test_embedding_store_roundtrip_and_resume(tmp_path):
    p = tmp_path / "m.parquet"
    store.save(p, ["a", "b"], np.ones((2, 3, 4)))
    store.save(p, ["b", "c"], np.zeros((2, 3, 4)))
    ids, X = store.load(p)
    assert ids == ["a", "b", "c"] and X.shape == (3, 3, 4)
    assert X[1].sum() == 0  # 'b' overwritten by the newer write
