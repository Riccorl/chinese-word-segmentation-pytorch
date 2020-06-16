import torch
import torch.nn as nn
import transformers as tr


class ChineseSegmenter(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        max_length: int = 200,
        language_model: str = "bert-base-chinese",
    ):
        super(ChineseSegmenter, self).__init__()
        config = tr.AutoConfig.from_pretrained(
            language_model, output_hidden_states=True
        )
        self.lmodel = tr.AutoModel.from_pretrained(language_model, config=config,)
        self.lstms = nn.LSTM(
            self.lmodel.config.hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0.4,
            bidirectional=True,
        )
        self.dropouts = nn.Dropout(0.6)
        self.output = nn.Linear(hidden_size * 2, 5)

    def forward(self, inputs, **kwargs):
        x = self.lmodel(
            inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
        )[2][-4:]
        x = torch.stack(x, dim=0).sum(dim=0)
        x, _ = self.lstms(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)
        return x
