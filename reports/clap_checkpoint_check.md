# CLAP checkpoint sanity check

88 FMA clips (8/genre, 11 genres); prompt "<genre> music"; one 10 s window per clip.

| checkpoint | revision | logit scale | mean audio-text cos | text-text cos | zero-shot acc | chance |
|---|---|---|---|---|---|---|
| laion/larger_clap_music | a0b4534 | 1.03 | -0.004 | 0.999 | 0.09 | 0.09 |
| laion/larger_clap_music_and_speech | 195c3a3 | 27.44 | 0.145 | 0.560 | 0.35 | 0.09 |

A healthy CLAP has a logit scale in the tens, clearly positive audio-text cosines and distinct text embeddings. `larger_clap_music` fails all three, so ReZonate uses `larger_clap_music_and_speech`.
