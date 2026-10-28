"""Standalone leakage-safe Pass-A NVARC runner for Experiment 015."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile


SOURCE_ARCHIVE = "UEsDBBQAAAAIANAAU12qu9ayfgQAAI4IAAALAAAAVVBTVFJFQU0ubWSdVk2P2zYQvetXEAiCXFZeS7JkeRsE2GZTNGgQBNm212hEjmzCFCmQlB3n1P/Qf9hf0iGltb1BDkUvu+LXm5k3j49+wT7+ef/5LXNSbxWmAzjHxsF5i9CzwZoDatAckyRlj2a0HJk2Hltj9nfs9c77wd3d3h6Px8UetgSw4Ka/5UbgbQ8n0HsL8O0WLE9hK9M89VYOc5RUH8K8Mx7Sokw3qWrfUJDfIsw5CHNq3N6x5n+gNT9Ae/9AWFmxWa6yYpU3l6I6qZCW/iv2Qg4nTRGSFy/Yu6/eAvfS6CT5fYfs08nvjGatERIdM90lOkelHCv/+evvmh3RIsPpKAp2QNuClz1rT8xibw7UjsRodWIIfBdPvnJMIQhaYM3Ll0crPYas2evwV0OPb5pLqJ6q4Heha+EoK6favigDAu1iODVPK9W8IjB07dnSel5yRh2er9S04jxYP08nHw0bwPMdFRwL6ykM84Z5miFyiePIsFswYkha9vjrfZqXFRNyi847BhYp2aZpPFGS5LzKy7wS0ELRCbERUGRrLARvq3oDnEON+WqZ1VhWuOLrLudFgSIrizpDUfOKsWe1JvkS+UqsRJF1ZZ6XvBPlSmQcctzk7brIs6xeLrlogVQBWYucRkVRc9yUdZZBMeFdGEryVVF3VbXJlpB1BbaiQ1hhVlX1cs1zOie6slitN8ss33QViLri2ObVsq6hzLMimwDPvCYdrrqszdfQrUuganLqc9lWGa6LrChbLtaIyxbqKs+pdABCCjFEBZQJrzaMXboRWIy6fK+5GgVpC7Qgpc0Dup+D0ai9S5KnHVEnwQQCZUFf4QSM2562QRR2OruEB7dPleGg2OfHD+bzPSMBSx3EStIw2sUhhXn45ZFxgpECPLItarQRKUJbdNzY6VATPvHLftsb3VwdcaiQz7FJRc88iSOZlIJRk+Bskrz7elVG4059a5QM/TdBnDGiJI2d7wZ+RT7GZIKY6cynYHo/x43x8y3N9aPycrLDS1KDIeSYdkgJBAxeHjAU7q1RipIhsBU1w47cjxZUKqTzUnOvQ8rXNVHqfD8Yqf0T4Z3URKux4Rb5ma3oGJOtTKzPhmZHTYwyNw6DCi4T6jNHHQyCfBhTBx3GWtL754g3iackevT2RGmrOZ+bSwLy29wna0aamy7wqHtys05SY+ljDBc5SuyD5KgdTgl+77UDJcLo0ii63C4Sdl4iZMo+TN0PZG/4BMTyxXIRwS7OOJlmoJiURRDE4gA2dIOb4WTldufZDuNVD+ZyDvLKJeedf2injN9RNgNSaM1PT4BX1MUUQjMIgb7dJJyom4ReQUEypbGmtGlSSEvUGXsiIh7GoGW2R6tRUdmcWhAmgn3ffG+hAdFGuyZyTdLEt+XLd3tgYp5slDw1QEm6tNaH6Ne+O2GFl8CjDrzRGA5GCvo3NyoNkWKnpQttPUqiIVZgcTBOhhLoYblKskmmCvCnmMLFu0nWP6YgPFgQZ/H8jiVPbUjjgzE/hhSTtECuQjD00IUDO3Dh1YCWfmkskn8BUEsDBBQAAAAIAPq4Ul0ZmrKQtAcAAMsYAAAOAAAAYXJjX2RlY29kZXIucHmVGNuOm0b03V8x5SWQeMk6q0itVUdK0uQpaqs0T7UsNIbBpuYWBhzvIv6958yFgQFvdnlYzw7nfj8kWVlUNSn4IpGn/cMbfSyT8JQy/V/eZOU9oZzk5WIRsZgcKT/SfcrcQ8M499YLAk/F6qbKSd2U8CKjpStOSyJhPInJw6JiAW8yico4vGd1zSpFRNwGacJrsiH4o+H8M03h5AIhhBN0OMC0nfg/LipyIEk+ICAJ4nMEOCPy1uFF2tRJkTs7r4e5AIwkuj3u+rMPwrnHJdludyDoEHOIur3d+bQsWR65B0u8rSvVc3noLUnhCUF5CEcUVjHRqu3GuBxszyJX/r8kJ3a/cVOa7SNKLmvBFUhW7MwqzjbfqoZJ3kUVsYpFQdHUZVMLIS7bm9VOsL4Ytruh1ywk6SwpehA3aRqUVbHPmjS4M47bU87SJGebO+W8JI8DQRt45qWPXt5qoBuw3p7RTAI4O8tjII6UnjaHIY2M0dzdTohxaUeBDl4RQQWY6JZrhJWiRsZXhtcwNue1Hcf41Si2sL2RHU+HrMgtgkOTpcy8fdQW4jDW++lq38yrPSfcT7UVSKDkAtzCQkyMgKaHokrqYyYCzwS0ZZnl4I2gslzsFlKeANIhiQJOMywk8kcJVFf3JqshF6VdKKdVRe8V6Hx6JzHC+3mUZOSXDXlDMHzggicPYN0Nue0vjrRkkFrkHbmzLlfy0kgwsNFnmnI25JYXNcqWcN7so/q+BE2AkDgtxYu8Zoe+7P2MGuJmSe565HcjakYvcPGO/PZMieIkT2rWm2uYlU8UB0lhtPYWHwYiSKdY0TR15zgOwZ/CUV1iiRN37BKysiafxA84em1DSvTFIkwp5+R9Ff7BwgIK3HohIEWUBShTELgQuvGSRLSG2lKDa4JxDohIAxBfQUDIqdP4dY+HManPFgUhRBRA7W3SetC8JqHdo8hUgMov+gH0IsXbxysT3dcNMk/pzyJnxhRpQaPAEk6ZhdfgKWgyTR7kNGMbxxmYRQVCwf2S1kdwc5RAn0MMy6llBdHuxs7Lly+BNVGsiLYDrUkr0DrHmwuG2/4yD+IkVSaWwSTPexrBr4HDWgiqYjUE6XAYuCbaxOz4/ID6hdOQ/+HfN5+BoatV/K+AHFQ2Afqeh0NRPCWAj2m/cpjy0cxuPNbQdhzSY1N6QwP+c0rKMskPBJKoagAZilMVkRbk6daknRhR2g0t9GpDVpNXYZHXSd6wxRhe2nmCgQ0Ygwi0gr8+L9Okdh3f8aBijgCxZllhl3ARLhh7hOaRoYW3OJWM4adWeK4WGAUJxLAIFOTAYJBlFYVKpJzjTZmooH6kCT1DrKui4TNXE3zIbUhJCmdX22cJZcLbxo7wcKtTsfNBhzbpHDGsCvlmDKaTZCTdMJq+QFBCKrbK452MJk5e45XC7ogmE1dFpnOVuBwjUSKDBTpvEHgqc3sSptiE4PoEqhgMFDVVdWZgWMdx/maVSh6MsTVuH3u8ikmTJ98bZkiQQ5VEMI0YCMVuKQKsPhqTnAsAD4smr0E1MXCQjFYHLBAxAkI7o5HIqp64eyx+EN7AyITvywLGjYR7PkhoyjqqMK7jGHT705Kc+5C2XQydMINhfxxMh6poSouUJiemunO/JUyjUCIPY+fKtgOLzM6zVpVeF2zKkVk73PHUytWEKWduybBfyebWEP3URRmg9VE5nFjElPsotRG68tKGyD0owtHsRgkLE5mHCStGEXGD49CKMOj+JIZ6C7UJBl+rIAqnbfcnTJwoCWtXhtVGCCZE8XTR4OLyDP/3Wmz601LJtpE/k+AXfEzgY+KOx2TdY6ez82YwGHuT6abdn9ZzWG4L9wfZ/JY6alS0dd7TI7MzQu9ZHh5Bv5Ml+iRvZU0RJeWDxjEyErMV+D50i0WPl9I9eGsc95DOAUw5Jx6UEH9l8/CQsikEhPWZ9RPN7Ri7qGlqXvXvsGWCRIEZecWWsluM0xcKr6i7eJBR+fNkHgctvJOlW5xgBQA2iqxumoEVlnNabzWhnUqeOSDxjcJwvIVoRV9ozh4suitvLJ+0+lZJhMS1ZXSxEOOmmXr9ioHUzKBgv58UKuAmRDAd11jwiqVEQg54Tla4CbTx3QB+uMNMMHDfxZV5sEXPbSJTwXAWKOXy59oG8nCN7N/2t/MjAiJDv6tAAgdaZXC5D3DxdCbALJU8xUIbsO8NTSecl+QZ3N5/+RJ8/Ovr108fv0254WOn0iNzzCR3dCcxV9PRE2vx04yCgxLLa2fqBzkNBFCLARYGIa1/v6x3F/tutetmCIFxDVfLOvNC6lmp1XjdujXqrn/138Yd9KNWx5i6aY3I67cwQG1bmRvdbm46H1cs4YHJsKaSi8OIb7mse92OKQw3AKWx5bj59YzQ80FDizSTbRY5qrSZIeVJjS21epJQtK6ThIL2KMVRTZfFjqvvZHOF0BuX8ZkOqcr45FPV2B44Xsv6ZwP6gZi9g8DerwFQzk5YM+c7vU1rbDBtB6Nle9Kd/GzEZpGuo+hZmt+7VrEQHx6WusCf9HdBvBXzwHY9/lqx87oZx1nSzAyKqGuTuavXs23rNGhykJ7yY+8JBXiUsgnEMIQQkaHy1l/FKsQV0voOlo8XrdiCXuDWsfgfUEsDBBQAAAAIAPq4Ul1ZQdm+Yw0AAGExAAANAAAAYXJjX2xvYWRlci5wee0ayW7ryPHur2j4QtLWY7zkMsLjQwZIJhggGSDA3BiCoMWWzBG3YVO2FY3+PVXVO0k9O8lDkEN0sCV2de1bV7Nq+m4Y2S+ia68q+b09NP2RFYK1/dV26Bo2DkUrtt3Q8EEwBfT9Yex+7va8rf7Bh6urq5Jv2aZrX/gw5ruhKvOxy8U4VO0uxJ8R+/SFwe/1FYPPyN9GlrDra/oFmNnQvbKqZQgqQfTChtc1rgCAXTA4bhNEGlbtGCJgFF1Nl6//3koqAx8PQ0vPY2SsDyPJdSXyl6IGjkVXH8aqa8PdgQsRrd1tlahaMRbthsvVFSgnbstiGIpjxIq2ZPQYHlUNSxL2QM+Kug7v2Gf2xj4n7PGOBHojOQlYPBc9V1yI58N2W/MyLIuxyOtKjD4DQA6sUHZN3POhOYwFcWqB47HD/1ooCcTzpivDYsVKLjYg8tgNKyCPRkp+KGrBFQ0HJZglRXVWEXFbIbd2N6u2UuFRXImy2lVAMCMUhRCAlQnwDRDCQRglCTEG3O94eH+nbFQAIZCpEKTCsJBPAX1BOkySR2tueNp2o2J8PeEWkQw7pOtRNZuRUJHGcbxyN0qmOajAklEiaAYeXPr/Pm1HSBcmLTLXvIVvtqEtc3CePPz1wIfjopWWHQL0az3BwR8ozAG7ZUEQ/9JVbdgUfQjG9PQC1rm62tSgCva3V97+AEFfjCOHuCVsyGOeV2015nkoeL1dsVEngbWfEyKrWASMDRzwPtrEobFumzEnYRVaKbiXNfBDqaVq+wPmjwv5hramd1l6TYDXmTWI0sb159+qBsCLYfztywGsDkkCFOMgv1UwvC1/++JCg2YqTAQj5hWP+4H3teaevs+5V+Qv8E2bgO/Ipz4hA8kYbKd0j99XDMw15pDGNs/gMxzCTAX3jD448gx2mlUFapYQp5/uM38Rn5rVtbvsR5KD6aeu5X5SNnkfPyYjEk4fw0ds/bZgZLO3O4zvbZYg0926eGwXHOVk2Tq/6yInh4+za1LHIqSoSlCKQ21dKHMYQzZIUtxFnmIWpP/RgpVr7vuI0HpUU7zlLX/NKSIFuZUTuLgqIE5LiVxmHQjbToTp490KSloGtWU89jyBmuEUX8LGkknkx7zddCUPpyxPyGTRjGnw1FAiRZHvLf/asnIRbStTrZOboFbXVQMZC1oIkTzeRX48OJg/swdf+Yr6kg9PJCs5SSYxUWg42hiOPtq6ajlqx+1FYtHXUE2xXfF98elIjGNdxibJaZZSKtRvkQ0iRIxCvTnVmVblggQQmZYb8ER+hKsFRTNiXxzV+TL4rKlv6dqC+4jJKrpiooHUjkX3UZzMGzPZbs0Z0UUUl21KetvwfvSBe4jNqW+RdXXR+37Y/BF6KsFHVfH+ILAwbho+PnelTcTd8FoMpW6v9vy4YpAecqykyc/DgftOVmB8U2zbeq+X0Txdj8YBLMoLgjiI0vt1tp6qhAFokgRDN353F6x9Jahmg9Z0P2V0UcNm2kq9fN8J7m5XW8Vr0RdvXKBIEN33izhiynLitRqfQ9NVRGvCMek6u37SbaIIWk1UMhw9XCKx6foj4rds4pMLAno7IQeqjbIJfHcDf9PwH9wwHNqvURBo76ICMX/qxh+bvuYNb6E9/tMwdEO4vf4RdSOwnesQNR9kbxecuv4csEO7b7vXNr6eZcPia64pFe55Jjz61p6ZrjHH/Qf+Cawlj/9jTkrqUd3+/330m/jowoEBG5iKC9kkw5fkdCY/FQn64QrTfjdUO++AqtwLobxWST6B0rgnb90rZxUE7ELaakSFW/EAO/U3BzvCS8uf9msNkO4zn8TZx6iEAYz620WMCmCGUTHdaoAJBaUXoKC++ctKE+oArrimp2EULTODTyaHNB14ZV7KMqhbeNtxPeNBPifE0qD4Ff/yPt/WxU7MLEdPAVFZbcZQm9eVSfGnEShVnac+RVvynCp1nmsZE9ei1qtcqyj/koze3BCRSCVRwjYt70PX5Nuq5uGmFsZj6Ynjqo6EGGoYHW3owkY4RNv6SdJ6HnJXlOEscoBkuLQlwRldXHdFKTSVaOUnYKVazGP+ihXfPI6sSRFnrlTlHGCNGFaCfsCWc3t9c3PD/gK7mG7NBOkM0oK78RzEsZsarJY89FJLzEbQgmYkeF8M4JoA4ahCrUWXwtGJOLXfCzwTO4veZnVE5S9vDvVYeapylANYINOBkxPdNNxDLosWKNnBmpyJYcft+jDwlwZ4jAvgJJS9EwRzM6fb4LQ/56cK8jMRX0liDnvZatHBTnbnmp0COpLDtzlvtABHP8nlmqUX2E+rLDtfYuLsM2GqgctE6hoTEQPCS/hM+nT3nBcd3gxbJxb8RnnGQ08mjxziRQ8RUIY3KsGKOX29kt5llyxtTAbe/UIaKWlSq3ZqHb3QQ810XI28gXJwSfMfwqWgL+Ai/2YsgYLM2CVcuES4FLT+ZIvGgj4tF1W7q7nKTfhge2g3k7G2KkGq4+0PI2TDtnaPizb3erXFZIqzG8rEGISxDUDi/c4LZT+3IzuwJZQNIQVgtTXsUjsiO0NZ4PQC2LgtGp7nVAbdYbtbrP0hvHMf4SpBjEM0BZ/kljtIKvGW5rphlVR+i0p8JqwumqeyYHhxsHYH2Jpj6Gy/xqoFAwTRZCBtVHwPhFCdW2M3uXt7fdrfneNT8GMge3FjSrkeBOcTUj5fzzwvVrG1n5wKjPD3Gdq5gcRyGtcsNEwCOpJohMMHEQwYCCUvHSbkC5nSxxWenVoctahAsGMYLnNUs2LceIujf7NjenpynEt55dKoQ+VCEiX1JUCqxRQHkvSqiJ5ezTPcLLnN+6dZuwL/vFj9SpCqBr9N7tFTi81ePUD7y6bRxrD+rXKp/okXF7K/cS0z6zjVVQ4aMESMdBunUVsRNodBKWOinxRWYpPGyTXUL+kEsJxpz5VPnDxBd2QgnT1ay5/JPOL9k1oHZ0Ev2zhNQjsZe0nWQ/xn6BEjVNJiJ2leSJeJkzCTxcyZuFl0OhfVofa+omYe4zufRnQDq7QfTSw3e66146M9b6zY1txM+RdVjpr0dYUBjf37ExuT/Gi7msgrF8fZfntJNd0vOzavsMy321siG5+wXZ/QpsGrU950/EzM3hIvt5IQ7JdfZBL9l2ReukKiYcisM8ETnHSYo7p+StQlFPFCGeSo7r8oe2iXShyOVyRFgn/cvkjQTbYy8gfsq3iSPSK6h+8YqvV2dIr5K/M8KkdZIfY+4FiQtyBkV3Qn0m1ze1JeGlQYJCYHzCdliM6UGzN+E4cmhIrbD5BJcdqFrwiEL5EU5uVxXk/AjvFLUR/gPKKAHhAIQOkHfX8wEAsTKcUHGeujfMxrDPrwwmhJJWJS0ZW35HdOuj9MpDlntxIzrc8rI6aeCzMMm4gvDjlopZHCGoALpckPjakMOqFdoDXLpKhtEI+OgTZsZ5dmpVjw8cR6e4p2zJSBSuF23U6QbQ54V5brI6eDyvVvWF7JIQgIMvNv6F76vDTXe9OBUOy3CXi7qKd0mfd0sR3HhcWWfCmYp7d1MuEq/sxJRanNV/zrcwU1Sgn7WW9xUsIlRUs9Jfhn4SZK1wyN0A3TJY/FDmUyXf90n80nvrO9+EF1JNQv8+M55m+n2ZsddsAgX4owRQ4+k/5ZIcyJE2zNPa5moCqwzY5FtpcIAGpAKPm0u/E2gd1CPwv7Th7S9IHWqq1xSDO8dIAe19k5yOYEl+yghkJh+JIuo37BGw5ZkeVlBKhMrUTeaZiUutzOK/uq2nzpdn/KpptO1Sj56G3Q0WROOlO/dgJLY3nXFxf5dCNxmZ/laQmdFDSbZnaaOHzZ44VDQkoxndDk/E3lKbq/sseFHCJ3OoT972Ua7CrnU7vlLhI//cX3tVpyM7pr8o7Q+GR2PFRy+2ZC3H261ovZkrtglgh0lsBXwMLgEx2uwz7G0hN9+S7SnWY0fTnMHN+juae5rqh+X3ZHBWDiz7wL8KLfAuiefuGbMUr7zBD9QBTO+6GLB+x5g+37ucPirL354JTwPUdXZ2mtPbcBPuzw7k05vDwlP29rAjPHYQ4N5+8fXK83foVrIf6xJiinx1t8UF7qc/y+xgOlW1tk6vEdWOd2dTV7nRERtPZMb6Rz+6wpUicTzI8jfjMvDk9NJfCy0txiiEM9zm5r9Luqzg1UooYKwV8PYmRPcPY8tAwvPWGtaota91KxPYlZasqh09M2wBah6cf8VN3e0wA7vVMTa+cs/6CGRflH7wC+el1B2Zvk9BODfMuqArU7elGQK4f7+fncLH31FZgPYJ5dHAV4cfRn3tI1crtzdYginuitJIkIdCRfYhNxHE/mIjryFaiOdz9JP4G98gq8kL60OC7Z644mn/Qm0LegA+HxFiPGcJUqJBm9aqUQTd6ZIuOu2I4mf+2hQdk49hb0aplFHGULbZyznk69B7PQznmvW2ueXooCGnN/X1T9N/L2TTdwYOhu0RBusly2htLSMNHSgkpoiCkVQW+5Ga3cB8Cx/vEQZMutFDbVqrDk/NdDUYeDqxh5iZQqNEsm8SW+Tdg9+x1V+5d5e6k/TwMv9rM4QgRX/wRQSwMEFAAAAAgA+rhSXa+Lze2AHAAAlF4AABMAAABudmFyY19hcmNfc29sdmVyLnB5zTztctu4dv/9FCgznZKORFuyk826q20dR866m9ip7b07t6qGQ0mQxBVFMiTlj/j6BfpcfbCecwCQAEjZTu92pppJDBLAwcH5BnDAeZ6u2SYp4rRcsmidpXnJTsOi/BQmi0244J/TGY877DfR4joPoyRKFsf5YrPmSVmYNTzfmSO8MJ8GcRrOeK5AHufTD2EZFrzssH+/5clpmq/DsoQOO7LFYqpKaaFKUapKZbTmqvwtjiaq/EeRJlWbNJ8u1UOyWWf3LCxYkgmcyq+ztcIGy+LtTCBVqBqJpKicpnHMp2WUJlX9jM/DTVzOomm5I+HeZ0CQap7JPZIEusjaPEyKOUyW58YYJwA6BISBDgalAVRFkThdLPBZ4pKU/K6EqSswOZ9FOaAXFOUs3QBd9Rc8zyV+GZ+XqsuClwE+B2scC9qFJQ9wLh1WbKuq0Jl866tiFk1XccUPmOOUT8LpamdHouzPoiKcxNxVz78fX56fnX/0dnZese6f9wNol5sEaTOPFps8RFYxN4xjlt4ABaIZIsHKZZ5uFkvGk5soTxOUW3YT5hFWFqxIoQEHSEW45vAUQ082DROWA2SoYZ/ed7Mc4CUsy6N1mN+zLCwKFiYzFrLbCIW84IDCTFV5PkD7IASlYIMBQck2kziaAjB2cOC/+5Gd/+X48sTE3P+TiQOyygKYtJvAzDpKdDswuaL0jnYY/G7YANTNB7lQ7Tx6n/NykyfU0r3xWDSHlkkKcgTkPU8TgOY4HuNxwRXYnZ2T048ADUXGJRhCkLIQ7Ar9BgIZB+YdfL74MPwUfDm+/sVBUCB+Ze51qBuIMshdztRP73bx23Xw4ewS++ytwsUi5ntRMuc5T6Y8gI7ZpiwMYHGah0HB+awF2KeLy+Pgajj8AD0O+x2YWyl7lWjLgnCzkF31XteXx2fnwfFvH1XXnt4zCaq+jfHO677Y7a3sR+1esbt3oJ3pmpd5BEI5YL3+OxCrrxucWiFgb9YBz9LpsmjOZfjl4uSXKxubmIc5WusA5Ivbs0cqvuHdNx02B1OtuvCbMNZmbg7yl+NP+sz75sxV1yZ251VXrZfkSMvM34JMTUF4Zuwm4rdi8msgKmjhpE0qPgNRv1xevAfg+36/mo8E/+H0ik036w2Y2+gG9BCBgEXgxTKNZwR7Ni+C2yiZpbdN2NA7+P3s/MPF70iuw31/v4b/Sup9ATY2F+OA6RESFBarYBpmbeheH1/9Gpwcf0Fu9fcNiBZIhMJcQQyWLcExCe0spmnOiUNBOp+b0K9OLi6HxKHg4vT0aniNVNEZJcAFk7CcLhuzHZ6AYgbvj69PUDEPVT/vTzfbN2+YC2YxBZN5c9BnMVhTEFT0YTdRgVa87x0hDbpEA9PAS9sOwvIaIJGZ5jFHAbpnr4XZ6RZQj8jzHAw39Vrx+wIt7DUYY3SmOTp3sPQcvMVtHpUA7oHYBg2P2MN0vhAlkOoNf3yE6ILV3Ls4Pz37GJyefRr+M4DkIZDyNs1XAC/MshjlGG1+jShEAPSGZhOBHQX/MQGDhVBxAuhMQCQhIABlfyUcRp6C5hfKvkKHOczmNsxnhc/YdTVjF8wx+7eri3MGwQOv5MYDeCAmMwTH4zArQJ0wiOqw2yUH+Bo+S0QIBkB5RaaADEIAtwSvB+NvcniOFkkYg3xKR7kEJICGG/CjiDh2EdOeaVPudqki52HcRVc3i8JFkhYlOMEi3eRTDswgQTiy7e2eYUb/SKNEpya5dGImQ16CBwpzGmYG0KKEGF9NLU5TmFO2+fYthmim8AEgun/X930MefhsAF5r5JgIOGMPZP4VRgaVBNbzKsMVZ3w+h0DrCKbP+Dor7zFgmAFuYGazNI2Zy++Ieuj991F7MKCYcLaIMI4I2Yez09Ph5fD8WswdJr0WcTY62T82RQnPORe6itEljIeRFqkOTm6epiVEI0nZXaMeQ39D1lHcQl0TJmlZxjzh09WfHWRIhTj9GPw6/OsVWBPXUcYP3XTtFvBJWXEs11bXEYbJ/DmmkcMehm/CFxbbWsFoouRA+EkxES5NAoElGZbCleEQBSt6ROS0qTyFLMISQ2CEHKN+QHMsQ2d89KNiHkEEjGUJXQusHh5loHFfV91GCCTjieiDa5f5kTGlPLwF7HDJ4+MU3LlAgt9NeVayIf0B9m8bDe3NQD2giKw6ENaBdgFcHwzguqKDnBpY4gQWAhCBuDcdiuw8EyGAOFqNEehqBZbyRkBFsAT3RkFFYKsVvjLF5VEPNgHYTi1O6Jda+CQ5iObmXlRVmumCUZD4pRjXVqCQmVSpsyy92UYmZcpMUjWRN0hF9TpMDZIgERqa1dioF69Q4G5UlURF9pSzlZ7Bnq9sJOesM1TWNJlajXiDpJYheIvEDz8NPw+vL/+qy7vABVytC55FwpTErEDZNH1KyqtOAD502uR97pNjdkniZ7CgL2hocPLOfybOc8KPS7GdHZzOXy5Ojt8jP6nO2XeOICQS5R6Ue7Lch3Jflg+gfCDLh1A+lOU3UH4jy2+h/FaWf4DyD7L8DsrvZPlHKP8oy//9XziYGvmnv0UQzyezv/2MbwHmo0D2+uLX4Tka0jiCxVeFvk9xCDDT2/ntangpmgVnHzBg7u0cX12dXV0fn18b7/s7X44/yPLBzvDiSpbf7Ahmglzhclcu+WHR5dZsJb9YL+GcMbn62rhh60abpqFrNKEGlbfEeH9U9dEWdbCS26NuxV6R5ukqSva+3vLkIDicBAsQ/6L3JijmZe/gxz19i2VvQqF07+1eT3MHFtw/DdALEbx+Eu640t4pqm5NGsO+WF5FPWJw5E5BgYSB9FFTHM+y0moxL1grd4LidGIOjG98/M+1Zrm7u6dDxxh9uskLiGQG1/mGa4PNhCUhzEA+cDvBnXr6NBxJGgdHnFki1ZzZDJ18uuJJ9I3nT05OrOTA00M8eApwztPyFALU2TDP09x1trBGhEhMbJvhWgSN2Rz7MfgHMbJBCbSBpDfgEyEIpaAD/UqHQYBS8HKwL3GTOLm4UelP8+lBH5v5sJSH4VxnU86772AiYMdER4/9I3N7+/1DtrsLCx8YZRrjFpPcWj2N7vhM7q+65nYrDEgjvmJnEKpztizLrDja21uAld3A2Ol6T+7thpEq7UXYtNjrHx682ZFLwjmE82uYIgefWxRuweN5R4WkNPeiI2eltlkGp2FcwGpid3cFa5JFobHFcRxC2YAJYWtYsiVwHDfd5Cz+qaAlPqzfElAiRohBb11iEBU/Die4J7mGqBdXLrB6QkbhRhSJkEP1BQmVwNaUEVENwikq/SzNXNWnFk/czdrSD0eqaiQB4DURyN3dFWANQZddNUxN0JvkNgcvDcGtEMGBmGg4hfUoz3Fn2BdNRL1L/3sGCBhlGRZhWeauBQ20JsCtXlRAdKxIIsl8opHV3K8aux4Elrc8dy0lI2IgEwdt/HAlRTpy1rCyWkbzMhBPwkgY4JqU/r4BnuSZgCJ7jBx8BudlRrMVPIppxT6m6rE/1vmoKIxg0MzGwEnHax0S//jUwPUaEqwzFjfy5OoeVOFn1tsKju0+09t2txJPOReKu021FXPFVpWVweMY60jiBBQ35hhOXSTx/afP7jNHFsoMoSGhU5gAd8KkGeF3IUIrjiiiGdHhyIiKsGYdd8SRCTJiVJQ5PY7HY491f7Ze1lQSe1cgKpsMZdXXhlSD1RxABxeJVU6y4C6sgutGNiOVvtMAI4f0OgCH4YxH0bjBW/xtCp7jYUmO7e6ga5L5tyCyMJAENmBGvOaBgAHCFNyZoIAXEcpn+Ry8ZrC3HagOCkxsCR7LQvl128CWviaz7diIsNLT9abi0UhZWaBfhx3hwqPb2983g3zgzx8d5tLYHRzLQ25xkHOUeO5+izK3QqyjkLF5JynIIbahxuwnbNloAgrxB3jbPmLea0KoSMZeD1i/tRqgYmWvtbJl1gTuCHrh7EXFqH5nKzAB2NmhfUo6tQIJxO0FkHc3jBd8An4fBf2ewRIwKbGMe4i4JzQJC1RGCMhYClpL72AJ8Aq3LbsURbHzT5/EfmZBm2ZrWEvOVPeTL7/h9lnJQ4jJ5vTqdpnGnN2k03CyicP83t8JqhUK8Dw4OT75ZSjWyWpdnk8DGgq1xp3xm2iqAkSIgFAGwZbK12JtqBrjgr0JvLF0r9uDazXdqg6KLIIvwgq3XlWBnSnvMz4Q1aDOCzwWQ2wGOlL4a0FmBIiMCbgcSA/26peCFPBykgazeeFKlxynEJSBbV6Hd0HCbwWZ5DPtdXXEvj68ytICT+mmSy7FJxA7tyj6olRvnlWGUprghFwHDuUXEDW7+96OPAnDd8G8rqaFiNvCBpOJsrVOH3G8j68raMHcB3Q4TIXjkbnb7XVqoKJXEmOY42r8QiaERSDZpGavs4hwPOhXXDKR8TF4dJMO69V8e80UcxcFWJC7zFUIopNZDxCxFecZls3IpKvNi156/jTbuJJ+xWY+h6AWZ6zlALhocmULY1FLlbLC8kCJZrnqPuBgcC1cGwQ6MpAUBLNXmkaxlmnLDhIV0VLHMVqfCsKYNoPcRggp2v9Ui2HTKKLa1ZZ+i9GU5EE/iaFlMnNdKdWjcux5jU5gquaWLjRDoVYy2fBL7/vpLF2h8Rrl4n4Qh+vJLGR3R3fg0STg22WEaQSgeT7+54LOaXoJxNNOEOlARGv5U6W1UkHxR1ZezdrguqgRqmDWYNwXxrDqDiY8XGPl/o4hLFumLjmIcY8xXY/OJpr01nFTpBbbSE0m6tiqtuDf918S7ivuwURM9uLqbH/bWCZe5ctQoqdmU5ui5NX1yL1B8gbBJjkPV3Ufe2loMkFFkwPDO+nzquycWJeJh6bTkoYPTVnPM089wHVEGLdr48w3EBcLKyldS8sgNpiwKPH4MxDRwkA4o5aNF0oRIkNq1kKQGVAnq86riZXwO4g4a7Nae0wDknCf1goJbfRAUtsXj6OjDgSXY7OlaWAG5mO312xMojKofXLTthYDXcQa1B/Av9cWYEEIha5FW2uIyqwMNM/fiMapQRUOGNW1LRrUxVYGoNEQc4lmHSZEHEyIwZfmEYLqqWIWaqhsWZQIOC3qrreDWAEjdXe/Y6j1SGEzbipr5WCqNrYfMEZQPkFGZ6o3BGf/KrQiSYNFHs7AIWK0VmcRNeK2LOc12CfDt9bwTG6oK923Y1ML+svUn0DatqY2L1Wp01RSSzE9O7ZpUUJLAV+mfE8qntlsq8pJdRvt+/tjtlvTUAW2dUvUO6taN4wv1D9N9zQPXtc/oXhPKJ1KqCsgZKw9+hbd2652ImKpfL8MYOixNXBR/cTAla7UY+oAPSPrT3TZpiqw4pxK4+d+3XBM2upA1FPc8ryQ8Sbu1svtYzkDMSw2v2+EPaJOQNhS2f4WQppFuSxMotIYCiEkKW4f2HhqdLVwqk8b5G491dfktNFstBcN6g5VS2Og1yYgK/zTmyrO6e88q70BS3UwXto9rDCqtY0kr2qEAWRlW5Xuwks0QOGda/QRDbJwNgMJs5lXLWyEu9Dw0c9usauiL9JrJGJQNASuGrnLGkg1xlXoi5feM9ZY7/lia/znmGM6T/F08VYrbNPUGuv2qmWQpPm6mk69+NVBqQXwdouUp7dkGwxp65iy2rI715TaTptkeg21E+KCmwjAx3YJlxCMlltEW8W+NVtDsRwygglj5C6zQjW7ttZU8c6KtVoERIu0KhsQ5njCohkNTeaMyXxXBKCTByQAE6sKY4fFlqWR4m9FqY6Jm7mF27XEq9l93DJZ0+F0bQR9EEzXk5sR7W4Hvcw6XPEAVzkiURV8DWV/oRIlQcbztZIwdPZSrhzH+Zinm8xMH8bMuJRJKD77HfNPBIhBn+1pUAaHrFxGuLEoVOu+5F1Qii4WjA1W0QfT8twRkCNPy967/XEHyFP+uI+P/R/w8bpz7Vd1VIRa+ov1nq8OOmHZi+YBWABR9wQzG1v3mJSOijYULIgYQBLGx56ahkmwdEyCPfwii6PSdXyHTgvkUwBPvbHdScNlJF9V4bZ4L/B5JXFQCdz3LM1nPD9i+wzlpMfEpFmfCUJ02AET8++wQ3bdYW+YIgx7yxS5OuwHpuhEwyyQq0UQorHapwxyWPkejGtJfcU2CWVvgGTW2ETyfEx2n2D3QxgIhoUhjO7bO0vBMe0ktCHn5SrMOtUgGgfI0QnqdZhiLSazNojcvsrCHQgSVNx46NPOUi2t+O6wucx6hcdt09KSVTx800WVWbLKfN9vgKIkUlAzmqiLUx5hUyr0xh54CSr21buDcduZjEZDU4KKUbgrpnfkhq97nnwYg8VVDSZVg4nW4CWbS6/Ye5kXi2n5Cefg6/nXDR2W4EGICh0xV06gJywDPVfCTDylpOcm/GIZZpzRCao4TTlCyd7fEyQGcfmF3bHfO0ID9gSh4e3v8PaXDqUWhxHlYK//LykvLOCA/Z0kb4Vt7TmCYqJvphG9Fuv8AsGgvqPoKHpd9x4bPkJ2kE5CpNy7MP6qg557YyzCSc0wl2iA50TqiDrn+SahvAcz7fHX448fPw2Ds6vg5OLzl+H12fXZxXlwObz87dyRIZ5Iw9LzJD8Mr4eXn8/Oz66uz04wTbLn0LYqphXWNlVPf6QX5Mcx8psBrHwdJRGmxQdhvEhzEEKwBCI4vA3x4D6J762Tiu05j/gTeY9Uwvt8GQRB6+pIonbTg/6bt3WwIsMAoMsmxjW38xXd9R84p1VVuqlKaVVa4D1B9bDJqiKsf5Pqga8nVViNz/E6WPJw5oz1bYU8BApky3Bw0Ldez/I0gyB4gHdl6kVKFBYDJ8F8jPolEhXXqhFPygAkZbrKUvD/4KtlvlI9fzCn6VrccxSXAKq7YjpaCDAvsMrayIzTeflV5iWTgKnVvvDpIuc8XzQJDwoViLhOpMZrAYgWi2qtBKz2ZtVcw6m645QmMCmeFXoz3MgWYMQFMjHh+kKZPmMQujXwUcDYb7ynOzjAiZ65iYOIUJA46OlMMm6gSTLrr/SBU5Dl9cAJZ+EaN4pARzTG3vJosSyBJNPw3pSDOA8KtPQgtxBFY5zsTCE+NcSivunRyuScY04kjNmQpyK84UAKRHVxj7VaHXFvS52897qtep713toCOZnDO1PG5sUsGzhav9ksC+YRWLhNAoI5E7qNNqSwoeG9ZnEDO0A2C0tp8PN5NZGyTDtz/Gu9BHvX+7Evq1TiMLxtSSYWRgivqbhzZ3QJZpo9oLF+HNeJl0fsoerx6DzVRegatAdGPiqbLDdpq20YQKRxe93He9Bgi7i8a2NtbVLm26BCQiP/Jo6R3OB6khYLQrcTQKUOJ1HZrJuG2BdsaW3A/zdmyiT+wHw02STzCJvTN299q33t3V3NPWgnqOKyML1G5y4WovhSFzfroghV+LRKpfvO+tm97ZyoJUgnoFo/+GXqil4qP9qrMspwHRQI/UdruvUOu5oYGJCsRLlHtjoSgJSsJrgHvLiiUrtgwV2G06XrGbcprE4qWH+Ux/4yLQ5gvSSTztwYRJEd1HumNdvjdYsmztVHE+RY1UcU3BZgmvZqR+yDGohvVrl1lFNFTs0kG0ooVBGUFdZUvQZ2ojwmAgMRaM9gL8ynXdDwb7zb3++/7eJjuIi6/T1ZCgjQdBnGKOUQHVLed212GwH/nzYwWvSNcKGN4WlEyozB0QqOO2f1Ry2EiaEE9gobJcFRHtSbg+SE5BV3eRkCYkrc7YAXhas1xsRJiAuDdCWjP2osUhHwxZFxVK2nHPyspRwYutdqVosyzf7BMRdV1rG2SN2iAJsSsoz0VtqMsMWkBYiWMDHQ0a1b1Bevtt/vMg2jzHXpJhluecr7KOqOobZQXEYl3cZG6wjiUw+5JTafbmahDw6NTE24CtZ8neb3ZGsKHWOa11Z71Fj2tBxl48+2L9M0u3e9Zrt2w2a2s5Db7hTADIiQEJyOyi43pUVdmoXuutz7oBqgF3iCVriUEmd1FIEm9WtevE1k6Kndyhxjvvg8JoByV/yJm7nbhlJF4B1Gcugh3crUDapShwnq4bdUnI46w7D8qjUhurIWpf5VCQq0OLtw6a7aZDM3P7kCeuvCS6/xIRZ6e7TT4KcIR9CYt130aF00E6MGW+SIYD7lVfQfet1A+a6BKmwBKmgrTN3AMHmUclzRPizEi4rcXosYq9FRcZANYDZ5PFMsaZ/281GQ/cO112DbV4vc3d16hdaCotdkFum+kjNwnvTXtgWEq1Q51fDpKx0iGTGYZwf9QFzKyNVJUIudiBXU79HyKrehVc0XU19+5sjKDdQsIV1xF8dU9oylZQRnCRFvKY/sqo7IKLsFaM/eHsPbTru7ZoZ1u3+qQT/YoB4/vxebrJK/jm2JvtuYt6Og4EshwCGFZYPokd8/wpKEKh7t8SvzF8DSvIwMg0j77+J9AOtP/GaEjYzc2Ne7iQ5Na1pdsh/rxtO8MG9vnFbQ1QHCyyynuAq3zXR2zaDSmlG9mf7E+U7LlOiV/uESx3Y6ZKMFv+tcHpR3t83yrhLcnaqyLB8em03qo5b6mBrvZLYamyJTEZcZ2xh5oq09MQO37vyzmGn1AYUxflTgJaGdIXXtMgx98AMA9BURENhq0CO/N39sCrUVEeq/OppCb719c9eM/55HsfraxYMkfEOh1K8tc1b/WedkAtp2isnlqNIDDHFFn069WtpOjpZcCj0hBIHLq0T4YZH2CWHSUHUg/0Qy2ndloQmtqT+zYetMk1ripArhqEwjXP5WuG2RfvxJatcnDRXAdg7hb7Lacji5nVHiYDfQcxe2NqYUK5iEIoygVUASoU3xiUnhL8zz8N5YNk/TBJYk6sgcLSa1kXkowXZBwR/ou4DYumRq++G3iqJkw59Gs0jjDX36peksIoEuSJBLQ6uzSLzZeyMOwrUF5rYfXpwWx8nuZAXE3GSw1l2HmUulToWAt03ENQooWMAJ3RI/Twvp94B1le3WAYwk4O0ChL/2E0P7t2WxDONoFuqRvXpAd2nKpfeU8dSmoiJqYyehPe63f7RWGk1W4/bg1/7J/LfBw2R11JAPWUlmb7LyHl8GUsYsAuRIcb+6hDh+AZjvJZL2VEVAFOzo9/FROMnwWd8ParuD8vLh/k9Co5egI5mz3dXZHWSG48s60EFzrVN0K5bOxHVKN9Z0zyuPhX2VkmgNVbvFF0OU09sOEeXy/iUQbVvSo4sodW6rhv/o6BCCUG148aKR6/rdY/afGPPwyBqTXvz9YwrJtqb+uonas4BbLW8L9KfdgWk6FV8fnh3dqV27c6T7+ed7CsuAS4ujBrIv6S5tHfRWxad7PT7hEeXNLMuBPK1htNaZfOv77/+jf9r8Zoq+hSyTtTrMuW39wlLbT3zcl763ZOEF4fD/x+V/FS/rywU7RwIn/6L98OHl5cUlgw7WEh8Pt1zu+QHtvwYBvuO2p68+hewT8ACwsLdV7B1nIuP3beDoXZ/OBaFpYj6I/uL7t3z0RuaXNCiurnpdbMqL+WdiE314p0WQtdbFfTJd5imaM33r4/sW0899HG37Vj/YLvyQZSGTikdzEqY5fcit8NHn4SG2pk70OY25T0gUqIR4IoEfIwscb/zS7Bxt1G79DQH6rNoDLfzBrFBw7iDf4MFKOfpyfHUVnB9/pq+y/YvT2Lx05Mc+HcxH22C2XVavCPEGoCOX79BAlhogFI7QRBUx50fRFl43vq3XYQ+P3qOWi92qW3NMdcKsPFIqpHRjIwJ1+H8AUEsDBBQAAAAIAOWhU12vYXTk7AYAAM0RAAAKAAAAc3RhcnRlci5webVYbW/jNhL+7l/B6r5IW1m2033J5uoF0mywCLrd7SXZAgfDEBhpbKuSSB1JOUmN/PebISVZcrxBccAZSExRnLeHMw+HzspKKsOkHmVupB+7oclKaMd/aim6eamSTfvA1briSncLjeIJ3PEkH6yOyrowWaVkAlpnYs24ZmU1Go1SWLFCJryI76XKQfmKizxk/6mhhpCBSGPyImQGtIkrbjYhKzktjNNMBWejEcOP1BGIbaakWHgX3z6ex39c3Vz98vky/nj5x9XF5Y23ZHOmjVMeOBnnlgYTowscvcPvbZaA7yVV7TWL/sFuQGW8yP4CZjbAaqELaTbMhTZBf5KNjSZRUmvmQtDMv+MaikwAu4MN32ayVkHYKLyrDROwBcXueWbYSir7gN+MsxR4yioFKRBOUkVWKFsxcpx9YNMzO0EfkoYU45rupzZZAUxI2s6IsIrgIdNG++3jnzIT/h6+kK085/KO9I9nT14QMC7SVvnP7P20Z9LChrsR6QKg8t8EgzeNzI9z9saBt1KyZGLLVRLTn5YFBdpkhbPrFt5nCKmsQPxNR9HNkHn3XkBZtNr7t4ruFfrge1/zdgMrlQnjr7zFNQHopJeYClyZH9o13BgoK6M7LB2O3fTP7MSCYkOnf36Ac21q7s13AgjBrJs16nGI4N/M83k3GsJ8p4Dn3Qw8JFAZdmm/MikIERjaOw6B84IliusN7pq/M48V+BBEcSx4CXH8dMZ28BT8sw2L7dr4EP5hSrQVH1lTMfrkH644wIA+TSKsk2dv1kmUyKKAxBzosbps3SZ1yiPy5jFOeLKBg4WHsDy3XnGtB5NYZN0GfpizkyMiR4FcZ1tigLqKXs65VAqwKWcpD/c2KznWi6U933CdB86i53nXsl5vWCK1QV3y4fEMg85BaCbRx6IgwDOBEWRENbe3t+xHXCwMPJiAVUWtMVs7Ayjj6A7ZC1lFprjZsjYVspBGVtMRu9gArxh5gMWEKg2bf2iW89SSmFG1SFCXttMl+jVyIGPJ6mwLzKfyKICjbJHlUDyiw0h9zJZ8GqDyrIgwMCtG4QsMyF8He4wVmFoJ1CFwmr1ivh0tpssAg5u5zbVhxw0UyOd16Vs91cLLBAbk2cXtlAuS5ohaK4aIUYwLz6rBeatTcUwP0rZ4LjfBk+bBn4Xs0Mr3VC57KiNk7zZ97QzacG8WFJoboo0JO3FittyfB2d6wSEsM4zQyjofTM8HlO+icmgOAHvFZm9RuG/mFTvFP3JnqGI0wmJoeYDN58yL45JUxV5z4NoTX6Gf7ekfnat1XYIwv9s3TeBuWcTTFE8A9973xmNkuzGxnYd0h6wzXxWSm5A15/B8Gk1fFM/hUY9XSNGtPJ7se2nPe1FYUA+iW0ks1J7dFwWlSkEdt5hQBeGrZCOxgdDzRTfjURZASiPr8PJFE4T/mBj/fwnMnZZjPC2PS09yvl4XMCG+QcKauPW6UYqadMhws1v19ivOhbwXZEb7DbkpQDKIS+QRXItpvgaDrZfv/Xr+6RM2XFc38cXX336/vL26vfr6Jb6+vP72peVFYljUFHUH2778uymXUr01jmkKaoE6y8flughtvUwSWVZgMmJ/PcH+Y4y0/BeMT6Ynb8f0yNfZ+GTSjGKrKNkgu4JYIytSu+s1tvX/wyBseVFTIYvnZlu0jnRy+6bguE/dOFJQFZza2Re8pLwcvm73at+S9RpvTz1ruVJuOJolvyOs4tRfHWw2FWtMud/rXDvdwxWHuulDL4mdc0t3OdFd3xaZsZPkx/LIfjXyrgx9WmXt+cG+X2iA7iU2nWa91D6/vojPP3+Of7389w0hhiD8gLs/84aupnBXr+PG4IH4x8tfvn3qFEzf//Sav05Pw5/e8unpu3en4fvT2Zt3szQJOX8NyQl/4wWRrooMKzs86LaOIGKnOiQ6N5bDnbAEZtncsdP3QKKnkObmBS/vUs7ys8N2hXBc5MsGRFudByYa3nvJRpMojpDbum+eMLISbwCvw3675+5ncSJrZLzGNuJc8hyQ9bTLpv6lwdZMLPP5rarBrSfMVjHhhJIFvsaFh4L90jpsXFFKQSm3MLyqPDO9insZ9nI3ajvRYeNo7yeglmxHh7MF68l1aCHbOYie2stmyCzq891+B54wx3omVt5dnWI2znfOz/aywcb9K00weTs9i2arJ0K+fw/ZdcOnlh3srQW3rKyi37jgazrwo3/RpL+HGd1uk3MfsJWMkGkoqv1auyPYKK/Bd+EFx0S+YAvd3t77G4Nu6IrfC7//Q0JoE2ruNzesQeSDnxMOdz9sUnLeeDI6toUH96znO2cdwn5aZPZ+ZTkPlJIKr1XHr1q9Ov/+jcoZ6tlpLdjbx38BUEsDBBQAAAAIAGe3Tl3HtxYxMAAAAC4AAAAWAAAAYXJjX3NvbHZlci9fX2luaXRfXy5weVNSUgpKLS1OTMpJVSgtyczJLMlMLVZIyy9ScAxyVkitKEgtysxNzSsp1lNSUuICAFBLAwQUAAAACABudVJdUo0bE6MJAAAIIAAAFgAAAGFyY19zb2x2ZXIvdHR0X2V2YWwucHm9Wd1v47gRf/dfwepJ6jleO5tkN8bmgPQ21x62KA6bw/XBMARaom129QWR2sRN8793ZkhKlCUn6T3UCGKJnBnOx2+GQzoIgr8L/o3vxJniW8EaLTOppVBsW9bs9utPTAulz7TMxZmuuSxksWPisRI1jBRazYIgmEy2dZmzON42uqlFHDOZV2WtGS+KUnMty0JNJnYsKauDe/6XKgv3XPMiLXMjqeJ6n8mNE/MrvJoJfahwfTt+Wxwmk8lfa5myG5ZJpVf0TxZ6vZ78+rfb+7tFfH939xlmF+/ni/eTr3e//3L3z7vP8W+391/iXz7fw8zThMEnmL+/vJpfnF8FU/t+cSkuF+dJ+355fnU9v1p07/zDJtmed+/bc3497+av5h8S8bGTd5VuL5KPl+379dX5tbjY+u+bJE3bd75IL8QW6Z/ByFRsmVTxd57JNN6BxSE8NmKJPojY2Y9sU5bZ0rAGwVcBcSjYw17ovagZkQI746wWiebFrsm4CS6KohAip9wyiBcQykIBVSLMIlPybcQADzi9YJ/A26IwkxG+vZ+bpfFTm7V/5pkSL0ldzdcvCob512U/yFTv2RHXxCPlWRa27J4KdflgV29nAX8kB6YidnNjZPdmURggUISJyLIIHQpQo5k5aoqj+H1NqUNvsmAorpWCEzCA46QuTUQ2vkpkEJ1Yc/Utlqkyeid7WFUUO6GWLJWJXildTzHqawOUpGwKvSRFbtj53AwqIVI35uWBmRSPSdakSKAEiVuz/7B/lIUAYvyaTghQlEo426LqnvQDEIEPN5lgiucVfD1IvS8bSGCRHTA5y4KpsqkTQfpC7vP6ACFORd0CzamAZsKqw7QEzDgadDNpJyDo7aixHVwvUw4FCoQoqAgiDcGmsPMZWIJWhv6CUeRgSb5jnwB4GC96+ZEg0AmOPPBxCQr8jmG7q+uyDgPDkjdKs42AP/0gRAHSEBGQeKxo8g2kX7n1NA168LRKm9o3+0pfIUYvmhnveqpMjYqRg8tepiLG4hyD96tGqxCRc4wSimV/qA3oT1CKGboWGUlrUSgo4JCNVPVZxjciA115sgcfo0mqzL57gayaTSYTgiyEAEv7LBWiwgfSxhiLsjBEHvVsB1EJcCKITlQJ4rJJ+lIUaHFSdytFlrbx4MRqxWPiSS1yzDwS3EkcroyEU3JaW5v6tRcJjAWyAM8HkafguJLoQqMkaUE6JmWhYUMFRUk0I1lUkYOuYtBSVVmFgYlyMKVs6KHI86vFxqaRWRpXe67EIt40RQpIelM5gfg2tF+PT7vatOzKgy0XugG0ro54PBlEj1v1et0B8C+oJiRoxWtAOHpgK2oBIfD0NLjEasl1WZ+VRXYwuFRHKDzWGPf2Z5o35MtxbToyxIg1kGDibPWR4uYJE4WvpsdsJztfjoDjizgYaGyDXCqFhbMVhrIcM2n1ZCU/e8CwSU+Vz66zsmTrF7Bt2Y62XifMvvZ2TZLTgz/+i0gxfEJTLX/L81o+bCFxDOhbO1GUOmmtrTDDmtdFoLU+8s3Hco4TK1NsoB7+ybQLVsDrmiLfu1ZNW/OlyrmGlD6lr4FkqxKoTtnppg0i/el+7XTKDbN8anltokNLXihQIu8is2QI7Gk3BRu9rilJcaJNvtuqykz5hw0Ka7uQu71mny86TjVsDQc4eKk0uyB7Na2Ek4M5EbS9N+kDgS201IcASgvPNylntsFdQb+0Wq4HrZOtSMRcl/p6PuREiJt2zmP+t6zCP5tecbk8W6yjI0GLjyOSSAekHqhhho9knH/437WJ1kaUJ2mfS/Tjm10y1OX7ixJGDfLZ01Psr9riCUlOCekl3mmJ4WvOj7q0s+uaeq7rw+Do0MFv1aJ8bYDsOlNR6bY6M65w6AWMb4Om+FaUD4Wfb0/tM9QERidXkOKOcQV0UArK2B9MXct/Mn2655EMcegcw7x7GsOyFTAKzvZxFHjt4yiu2sdRwLSPr4V1xJdT5ygv0P+3GMd4+ot5kjQ1Tw5hVQt7FqIDOwa40XsTbYruNiu5Xp6utJ2AaGD7fDY3fik1zwAVOX/sNm/V5GF7qPXSxxM4fZWYtLV0xoW0+WHPckNM3bbXisUKs14lsG3mxRqP0ySkNzp2Kgaf7kSYy4IU8bU0OzlpEvUP1EbcS9y0rC/CDFg5vV22teydcag7mydlLWLo2FOJAo3FqtlQ74ZhfXMz7bWetnU+cTq7xyXZF77bZeJM6QMctXVZnemHknENp4IK+jV39haPVUk9pGkOWFOBPMHzdv+mTqMWIJJO3CvjfIOwVtqNMdkfEI880TEeQrpp9+ZgVy0I7p5cr5Oe0mVhAt29a9l6rfGMhIUeqm3UPPobz9F06mpFr9b9g9LQQqeNLFLxaPMOFRBwNsedQITH6h01hHRcuxlqtSKJa0xXemKfmIc5T5y5t7BHDPfxPNzfBt2S5nRpyeIFVBhcaPyISguYm5vXRZ3/EVHr3pvp6UDzfo2yK5jKYV/Q1c7WvhACFgjhxSEcqm0Fm1s3K8rVkAG1t960Y7Xtg1t9atSOetzR0K4O/D+Y2jbCdpQlP5gDhXuPhnbalAFCWeBlFAyNSWypFkeTLsNmvKpEkYZH24tbma5UbYkcgNjlhpMxcOPTYAQ/dPyJCeLB0qbROGHffUDcee8EBxh2HpM3gJq+TxMu+jYDQ+eWVa9tdp/nE6H2S+GoM4aOCGzFwVVt7RnSOEPwJj70PR4NaN1lsiFY+Y6wTS5NIIZ7gkZWJcxgZDy6Plnnh95G5zWPZBMdbLFJtwfmkUUNobMTw+sMGNf9lBjS1a3nQf+YpAeNAUlXemIs5EB3lMDvjhMV72/6I1TvoIHybRzDGtra4S0C0eSmbqR3R9QOt6Nj64BzlANUL3Du956kzCvudx3quO3oNRb9NuLNLQhSaQk9025w5zf6u8CJbsXcJtAVMc+F60MQ6LnQtUwUuIUJaMgP3q+H0K+2xrWtygl8ypRqir3ld2b5sOr8BIT9JC5Ap+WgiescOe0c1c9WBDQyTz2vUwPTxcC1MC3fsx9l41mMs3nCK7an536gH2oQ8fJtbYw/iS7pl9CjwA5mnLcGE6YjiVNZu1EKKMZ2ebQchBx/m53BsST179hI5gw6yzTW4lGHdp9plenz9XUcsjlN+1w9/YdM/WswYB276+409iI7bReMjvwxy7/B/xCzrdDq5rcaf/QUjxLvGb/Rq+EIOxYoAYFds1tshmYE0cwElLRuoUAWpk1eqdCZYC++y1rdhME0mLJgGfTOJOPrdc3z25Yznnpluf8CUEsDBBQAAAAIANOhU12zkbRCXQsAAEssAAAYAAAAYXJjX3NvbHZlci9udmFyY19ldmFsLnB5rVrdc9u4EX/XX4GyL2Ij62L3qZpTbjzXtHPTaSaTSa8PqoaGSEhmTZE8EEysuPrfu7sAQYAEHTk5PST82G/s/nYBOoqin9//66oqixO7F0UtZMP2lWSF4A/8IK4avhfs3a+3H35m4hMvWq7yqmS8zNg+L3mRf6EHyyiKZrO9rI4sSfataqVIEpYf60oqIC4rRWTNbGae/bepyu5aCs1Zc3Vf5LuO7T3c6hfqVOfloXt+W54W7GdeFHxXCKOUyzRpquKTkEulVIKWduS7Ni+ypL7njbhOdm2ZFWLB7vNMJEo0KqlaVbeqWbC8SYArz5KDzLPZbJaJPfsscyWM4CS9B52iPIhmPmPw6+8TtHxFBi/oleLNQ5JnzYoVeaM2jZJb/cKIyrjiSZbLjidmV2/ockVUEMx/o2ZGqwJ+iuJqL4VwVLK8VBVT98KIvJKCZxgQBmJFqip5ojXxDWVrCvyyqHjWzAcOLFEEBOVRzeOYGOt2V+QpMD2d6R7zwrgG+nsv6SX+8r19D0uONL2Ongp/kueNYP8Qp7dSVnK+j4550+AiWwbS9mTEnaPYsmurNubNFswbrabjmqWLQwuwPD7Av/OaS1GqZv1RtpAc4jFHUQ90q9m0XFA14Gc/sKjXtcTgRg7DUicQxZQCn7XHuplrDxasEaCYw1o163m0iBYsWkWxib0UUESlkWPSEZctMWUHRlBifD0ZweSWiu/yNO11DJM0y1MiXGAZbm26/pMrIXPiIWxAS3XiNozv4Z2TqgxKEWOsRPb9KWqd8/l8n8dsyaKzbh0CCEfvolexsAGLx1G6NJNILVkFqj0JmEvaKCePevKJXNIcF+WSEx9HrBcbnWdJXfC8JCCcI+aLFa42rT8lCv0DALTdsv+xd1UpdB5A9UMQuVJSc4ENqkLaKO6Ln96A6/T/Ur+fxx0/oYYLxFqSw298Qa2ubxu0Z56KoogJOPAK8UdWn7f0AC7wnsRtjZ9d+BNV1TfDOloNct0UkygAXUXmvqVwIMnWFIlq60JsBuwOA4XO1s7fjBWsSStomw+HIzRYrYcS+3MOuUIRARkCaumYl6ARgHkPtu54+tDYOmraHQEpSLConVYtZCTe2yBGKRRpDiAmDGImkFrHJlqx14ueCFC1VLk6JVbPgAAStwq9DHaLBpqxyJzKchaVtIOFm619hNx5mYlHqDpEdiRBMaJsjwISXYQwfhMhbbSN/WZjvUUd3QouD0JB37EdJnkidWconc029vgp+L55nYlWNDU7q8fXjz9MZhDhlpYlj0fUUArEkDdUEpjthKv00LRWsmqsyBq85HUtymyOPEENELw5UcZsvWY3YVE7wIcHP5qUT5upHNqyV2vM8Pmuqgoj31cPqidsB+wQx1ol1xog8mzz2g85QJ14ns0NsE2cTZSXYGG0DcahZ4dw93A2/OmR5VcEkG5oyUudGqiIkQp/atmYnNpGY8VdGANVRiG8HsbMWa437Ho6Cjc2eNcvCR6ybSDe20lLB+UeshIzoMu7p8gGFqDBXkNXsCqd5zfnPkQ9irlTHgl3Mb8nWxgbu/6F6J/wNG0lT099na2G7QuwRbY4Ew2eE4rvoVUq29kw+n3Bsj+s6QmxxwzWnJenuXXAI06g9fgM9MRDOo/YWJWYlvUlr3th5p3mHrfF18vXGvMhAAR27XGOaskEpw06QkzXJY5peZXEXYWRaKl6s6nfAopoy/Hud/BvzI2CF46SEf9AvomTmzWdKz9ol03KcOjBJ5gEHFCD3Cm6vV4m0ioTWSJF0xbKmw1CVzQMmIlBD2hT8wKlO/Z96ZF029vN5hnxjqDt8+P5rfYOIKx8gE1WN5ObEZiGdZwyAMfZscry/Yl2Yra12WFkOSOJd3eDeNzdIXTiiE/AAzwZa/gRpiBWQxdgtayyNoWHuxMw38r0r8Qv7+6WJPAtT+9NGGCZUpF/glYNITCdV4vCPXqZFm2GxklRC0cNKueQxFzKE0n8VCk6Mighf4509rCgBnpsAakxDEzAVuTE2jL/DebR3ldsHMsubl2BNDAqmFWiSn6KnFkNZ219u28BdsDX3bEtkj9HZ6echr0j2lUw1elzFSOZwdYB/PqthQ18Bg1j1iEqRBlPMpwBiVBx9DQ8cOlFdqBCk4QnLt0x9dgl+eeECskfu7S8flvtdxTN4LdhK2g0BhjxU133uY5LnDQODTpu78Ko7T6IE2663JnPoY482i6v1sPap7kRJC1gvg6MiUnP+DRySL8DzStzOaJAZyzVoisiXE0tdkkLNw/OMW7MNTnZGnUbWNgPugOlJ+Ls3ZmqCE68vUmev0vaWYFlF42+mmnTmxaYzHjTCKlGc/DkrGzGYm17eITT7/zBeHAspQFyBNeDAWG77fdXLq/HCJRmfxygplBS4SclP0LfM/CHS63xoFvqUF2UDyKwIenkepsSLW3uLVZsZ3Cbrjgcgrxw3F62eXEX5bkNjHYjrLF38/ltzB/ZbRc5gnbxWFfgiAZ3DRVDiF9O7YW0PjuqaT473I0tp1hrL7u0CyyW9mQAZJM+Azg5WXFm1TFX2OacLSzhHQDQecWiZ+RoG9dPjmfnhTFTPzUmn8NSxrHuimPjWIiZraUHyUe0Yc/NShMOdzP1K3bttAgbc8rTIA4MShm6fnoP5ibNPa8Jy4LAFCwgf1Gn8oUyEofeficwBjHIeRhkwm4HtgnryW3C0MavjdRo3Wia7mPl3vkoYvZNOfWwzWAv5UoNhMqXtLPNFSQd+eM4CrCoBzF3kjFeYJdeF/y4y7g+AFo5Fm3oCR6miT2HbrweZUI8tqCHw/U4c2AtHTMduBpXMtEZKNTmbnrWMRKPdI/nAfxFXoAjz1tH/CLMjOYAjzVtgswZVxMsJGCx1ezNsltThFYgFeKzUv2ptxM/DXFDxYOpOWxBGKLGdvlNVhdFDZObKHHzasEkUNh0wBhapMhMi+CUuRqrjfo5Esn6KXpMqVU7G006ugAupwoCbD6cTbD7RCExXkTQVPc+QO8mjoGYPn2GuTPJP0iRgKDLE2zcrab1Grur+nrCYsTbQHYP2W8mEzoo9LlpFX/YFIKMP67ZzRirLw6r9vMCQ4clR2H4LsUXRmik+MWhGkm4OGY+HidD2A1bHwbj6a6h55Pu2OxlJnU43h+GPbNF63fx3YCMN/EI0rpjAp/KPwwYkHgyvhEPSbULTb29oUDow7ZEPPIUGXBi0uf2Pl7pT3ndp6DnRQ5L2BMe3rOQxlHpTyn9vnp5mTmhgvsuu7rvfT3bVLK5B1G0oEyPqG5y6bQbdls8Kv76Ko7l8MNBisNwZvJTqr9xvkeSM6Ok62S7hK499tvnRC+MDqLEo68cEgK/N1h6JyRXk8zDRmS5vYP04Xo7jWs6Wv0STeu7eaG+YVq9SJ89J/bjNKF5mBQI9bS/nrBrdHzyQhsD3fKy4ITa7DdFKQQD32jBN67TQNDvv2BBS7995dy+kGA1d+bh9cZvG1t7AG7xMFD64bbgyR712XE2flXlC7PgBUZM95LLrdHobj6IPUUWcLEdddcLjae2SRhR59lslnz8cPvLu1/e/T358BbPf8QyrY51XghttYRM4XmJ318axZU+sarbL18KweY/vf8Rxb3ZvL76C7/ab1/Fq+WffooM46GodrxIGiXqNZLiRfPmP9mr2CHqpCdF1TREhhdvNlevQOZSvAWZ0az7W6aay0YkloUMmhfVgf7kCUKt5ODD2eAzm/2G9vYRhKSK1UJeUYw/fvzIyL7+r2EZGsLoL1LbGmQIfmRpVTYVeA46+7/T6UJvF4Z2bsuDrNp6TnGP4tVg9orwL1tJIfaeUs09Hv0iHvR64kGjgIW+Z/tM9Mbl6ds+rhnRYi65C74ETzMoXGmjGJuc+j9QSwECFAMUAAAACADQAFNdqrvWsn4EAACOCAAACwAAAAAAAAAAAAAApIEAAAAAVVBTVFJFQU0ubWRQSwECFAMUAAAACAD6uFJdGZqykLQHAADLGAAADgAAAAAAAAAAAAAApIGnBAAAYXJjX2RlY29kZXIucHlQSwECFAMUAAAACAD6uFJdWUHZvmMNAABhMQAADQAAAAAAAAAAAAAApIGHDAAAYXJjX2xvYWRlci5weVBLAQIUAxQAAAAIAPq4Ul2vi83tgBwAAJReAAATAAAAAAAAAAAAAACkgRUaAABudmFyY19hcmNfc29sdmVyLnB5UEsBAhQDFAAAAAgA5aFTXa9hdOTsBgAAzREAAAoAAAAAAAAAAAAAAIABxjYAAHN0YXJ0ZXIucHlQSwECFAMUAAAACABnt05dx7cWMTAAAAAuAAAAFgAAAAAAAAAAAAAApIHaPQAAYXJjX3NvbHZlci9fX2luaXRfXy5weVBLAQIUAxQAAAAIAG51Ul1SjRsTowkAAAggAAAWAAAAAAAAAAAAAACkgT4+AABhcmNfc29sdmVyL3R0dF9ldmFsLnB5UEsBAhQDFAAAAAgA06FTXbORtEJdCwAASywAABgAAAAAAAAAAAAAAKSBFUgAAGFyY19zb2x2ZXIvbnZhcmNfZXZhbC5weVBLBQYAAAAACAAIAPcBAACoUwAAAAA="
RUN_CONFIG = json.loads(r"""{"mode":"single_pass_candidate_diagnostics20","task_ids":["0934a4d8","135a2760","136b0064","13e47133","142ca369","16b78196","16de56c4","1818057f","195c6913","1ae2feb7","20270e3b","20a9e565","21897d95","221dfab4","247ef758","269e22fb","271d71e2","36a08778","981571dc","aa4ec2a5"],"n_train_aug":16,"n_eval_aug":2,"task_cap_seconds":1200,"lora_seed":42,"train_aug_seed":1,"eval_aug_seed":2,"score_seed_offset":0,"order":"sorted","nprocs":4,"smoke_deadline_seconds":14400,"submission_deadline_seconds":42600,"selection":"score_kgmon_top2","lora":{"type":"RSLoRA","rank":256},"report_filename":"candidate_diagnostics20_report.json","upstream_notebook_id":139041342,"kernel_source":"sorokin/pip-install-unsloth-flash-patch","model_source":"sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1","competition_source":"arc-prize-2026-arc-agi-2"}""")
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/experiment_015")


def extract_sources() -> Path:
    source_root = Path("/tmp/experiment_015_source")
    source_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
        archive.extractall(source_root)
    return source_root


def find_arc_file(fragment: str) -> Path:
    matches = sorted(path for path in INPUT_ROOT.rglob("*.json") if fragment in path.name)
    if not matches:
        raise FileNotFoundError(f"ARC file containing {fragment!r} not found")
    return matches[0]


def find_package_root(package: str) -> Path:
    search_roots = [INPUT_ROOT, Path("/kaggle/usr/lib")]
    matches = sorted(
        match
        for root in search_roots
        if root.exists()
        for match in root.rglob(f"{package}/__init__.py")
    )
    if not matches:
        package_matches = sorted(
            str(path)
            for root in search_roots
            if root.exists()
            for path in root.rglob("*")
            if package.lower() in path.name.lower()
        )
        mounts = {
            str(root): sorted(str(path) for path in root.iterdir())
            for root in search_roots
            if root.exists()
        }
        raise FileNotFoundError(
            f"offline package {package!r} not found; "
            f"mounts={mounts}; package_matches={package_matches[:50]}"
        )
    return matches[0].parent.parent


class MemoryMonitor:
    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(1):
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                values = [int(line) for line in output.splitlines() if line.strip()]
                self.peak_mib = max([self.peak_mib, *values])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    source_root = extract_sources()
    dependency_root = find_package_root("unsloth")
    sys.path.insert(0, str(source_root))
    sys.path.insert(1, str(dependency_root))
    from arc_loader import ArcDataset
    from arc_decoder import ArcDecoder, score_full_probmul_3, score_kgmon
    from arc_solver.nvarc_eval import (
        analyze_candidate_oracles,
        finalize_top2,
        load_finalizer_labels,
        parse_training_stats,
        write_solver_challenges,
    )
    from arc_solver.ttt_eval import score_condition

    rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    challenge_fragment = "test_challenges" if rerun else "evaluation_challenges"
    challenge_source = find_arc_file(challenge_fragment)
    challenges = json.loads(challenge_source.read_text())
    task_ids = (
        sorted(challenges)
        if rerun
        else [task for task in RUN_CONFIG["task_ids"] if task in challenges]
    )
    if not task_ids:
        raise RuntimeError("no configured NVARC tasks were found")

    # This directory is the only data directory passed to the solver subprocess.
    # It is created before any solution file is opened or any labels are written.
    solver_data_dir = WORK_ROOT / "solver_input"
    public_path = write_solver_challenges(
        challenge_source, task_ids, solver_data_dir
    )
    public_challenges = json.loads(public_path.read_text())
    output_dir = WORK_ROOT / "pass_a_candidates"
    telemetry_path = WORK_ROOT / "pass_a_telemetry.jsonl"
    log_path = WORK_ROOT / "pass_a.log"
    marker_dir = WORK_ROOT / "markers"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(source_root), str(dependency_root))),
            "ARC_OUT_DIR": str(output_dir),
            "ARC_TELEMETRY_FILE": str(telemetry_path),
            "ARC_PASS_NAME": "pass_a",
            "ARC_LORA_SEED": str(RUN_CONFIG["lora_seed"]),
            "ARC_TRAIN_AUG_SEED": str(RUN_CONFIG["train_aug_seed"]),
            "ARC_N_TRAIN_AUG": str(RUN_CONFIG["n_train_aug"]),
            "ARC_EVAL_AUG_SEED": str(RUN_CONFIG["eval_aug_seed"]),
            "ARC_N_EVAL_AUG": str(RUN_CONFIG["n_eval_aug"]),
            "ARC_TASK_CAP": str(RUN_CONFIG["task_cap_seconds"]),
            "ARC_SCORE_SEED_OFFSET": str(RUN_CONFIG["score_seed_offset"]),
            "ARC_DEBUG_KEYS": ",".join(task_ids),
            "ARC_ALL_KEYS": "1",
            "ARC_DETERMINISTIC": "1",
            "UNSLOTH_DISABLE_STATISTICS": "1",
            "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
            "OMP_NUM_THREADS": "12",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    budget = (
        RUN_CONFIG["submission_deadline_seconds"]
        if rerun
        else RUN_CONFIG["smoke_deadline_seconds"]
    )
    started = time.time()
    end_time = started + budget
    command = [
        sys.executable,
        str(source_root / "starter.py"),
        "--end-time",
        str(end_time),
        "--order",
        RUN_CONFIG["order"],
        "--test-path",
        str(public_path),
        "--marker-dir",
        str(marker_dir),
        "--nprocs",
        str(RUN_CONFIG["nprocs"]),
    ]
    with log_path.open("w") as log, MemoryMonitor() as monitor:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    elapsed = time.time() - started
    if return_code:
        raise RuntimeError(f"Pass A failed with exit code {return_code}")

    # Parent-only finalization starts after every solver worker has exited.
    dataset = ArcDataset.from_file(str(public_path))
    decoder = ArcDecoder(dataset.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results(str(output_dir))
    candidate_stats = decoder.candidate_stats()
    selected = decoder.run_selection_algo(score_kgmon)
    submission, fallback_counts = finalize_top2(public_challenges, selected)
    submission_path = Path("/kaggle/working/submission.json")
    submission_path.write_text(json.dumps(submission, separators=(",", ":")))
    (WORK_ROOT / "submission.json").write_text(
        json.dumps(submission, separators=(",", ":"))
    )

    expected_items = {
        task: len(public_challenges[task]["test"]) for task in task_ids
    }
    actual_items = {task: len(submission.get(task, [])) for task in task_ids}
    if actual_items != expected_items:
        raise AssertionError(
            f"submission item counts differ: expected={expected_items}, actual={actual_items}"
        )

    task_telemetry = {record["task"]: record for record in read_jsonl(telemetry_path)}
    training_stats = parse_training_stats(log_path.read_text())
    for task in task_ids:
        record = task_telemetry.setdefault(task, {"task": task})
        record.update(training_stats.get(task, {}))
        output_stats = {
            key.rsplit("_", 1)[-1]: value
            for key, value in candidate_stats.items()
            if key.startswith(task + "_")
        }
        record["outputs"] = output_stats
        record["unique_candidates"] = sum(
            value["unique"] for value in output_stats.values()
        )
        record["candidate_samples"] = sum(
            value["samples"] for value in output_stats.values()
        )

    report = {
        "mode": "competition_rerun" if rerun else RUN_CONFIG["mode"],
        "task_ids": task_ids,
        "config": RUN_CONFIG,
        "timing": {
            "total_seconds": round(elapsed, 3),
            "deadline_seconds": budget,
        },
        "peak_gpu_memory_mib": monitor.peak_mib,
        "fallbacks": fallback_counts,
        "tasks": [task_telemetry[task] for task in task_ids],
        "submission_item_counts": actual_items,
    }
    if not rerun:
        # Solutions and labels are first opened here, after the subprocess exits.
        solution_source = find_arc_file("evaluation_solutions")
        labels = load_finalizer_labels(
            challenge_source,
            solution_source,
            task_ids,
            WORK_ROOT / "parent_finalizer_only",
        )
        label_items = {task: len(labels[task]) for task in task_ids}
        if actual_items != label_items:
            raise AssertionError(
                f"submission/label item counts differ: "
                f"labels={label_items}, submission={actual_items}"
            )
        report["label_item_counts"] = label_items
        report["score_condition"] = score_condition(submission, labels)
        report["candidate_diagnostics"] = analyze_candidate_oracles(
            decoder.decoded_results,
            labels,
            {
                "score_kgmon": score_kgmon,
                "score_full_probmul_3": score_full_probmul_3,
            },
        )
    report_filename = RUN_CONFIG.get("report_filename", "smoke_report.json")
    (Path("/kaggle/working") / report_filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (WORK_ROOT / report_filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
