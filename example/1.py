class simple_bert(nn.Module):
    def __init__(self,cfg,config_path=None,pretrained_path=None,dropout_rate=0.2,fc_hidden_size=768,use_keras_init=False,use_pooling=True):
        super(simple_bert,self).__init__()
        self.cfg=cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model,output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained_path is None:
            self.pretrained_model = AutoModel.from_pretrained(cfg.model,config=self.config)
        else:
            self.pretrained_model = AutoModel.from_pretrained(pretrained_path)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(fc_hidden_size,1)
        if use_keras_init:
          self._init_weights(self.fc)
        self.use_pooling=use_pooling
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data=torch.nn.init.xavier_uniform(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
                    
    def forward(self,inputs): ##直接传入字典不需要其它的
        # inputs的输入为x，不要包含标签
        #dt=inputs.get('inputs')
        for key in inputs.keys():
          inputs[key].squeeze_(1)
        
        #outputs=self.pretrained_model(inputs.get('input_ids').squeeze(1),inputs.get('attention_mask').squeeze(1),inputs.get('token_type_ids').squeeze(1),position_ids=inputs.get('position_ids'),return_dict=True)
        outputs=self.pretrained_model(**inputs,return_dict=True)
        if self.use_pooling:
          preds=self.fc(self.fc_dropout(torch.mean(outputs.get('last_hidden_state'),1))) ## use only cls hidden states
        else:
          preds=self.fc(self.fc_dropout(outputs.get('pooler_output')))
        return preds