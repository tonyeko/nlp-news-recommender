# Download saved model from https://drive.google.com/drive/folders/1-hQRq3lBpCdsOZU2ln9jqlnaPBw6L5GH?usp=sharing
# Then add this saved_model folder into keyword_extraction folder

from utils import get_text_keywords

if __name__ == "__main__":
    text = "UK economic growth slows as supply problems hit the recovery. UK economic growth slowed between July and September as supply chain problems hindered the recovery, latest official figures show. The Office for National Statistics said consumer spending increased as Britain continued to emerge from lockdown. But that was offset by falls in other areas of the economy, leaving growth for the three months at 1.3%. It means the economy is 2.1% smaller than in the final three months of 2019, before the coronavirus pandemic hit. Sterling fell to its lowest level of 2021 against the dollar on Thursday following the news. Grant Fitzner, ONS chief economist, said service growth expanded, helped by house buyers rushing to do deals before the end of the stamp duty holiday."

    keywords = get_text_keywords(text, 5)

    print(keywords)

