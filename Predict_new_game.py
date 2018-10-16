
# coding: utf-8

import pandas as pd
from sqlalchemy import create_engine
from urllib.request import urlopen
import json
from datetime import datetime,timedelta


#crawl當天gameid list
url = 'https://tw.global.nba.com/stats2/season/schedule.json?countryCode=TW&days=7&locale=zh_TW&tz=%2B8'
url_player_data = urlopen(url)
resp = json.loads(url_player_data.read().decode('utf-8'))
gameid_list = list()
for i in range(0,len(resp['payload']['dates'][1]['games'])):
    gameid_list.append(resp['payload']['dates'][1]['games'][i]['profile']['gameId'])


#Load model data
#球員raw data
engine = create_engine("mysql+pymysql://cb102:iiicb102@10.120.23.10:3306")
engine.execute("USE nba;")
player_box = pd.read_sql(
    '''SELECT * FROM nba.playerdata2;''',engine)

#球員近兩年表現
gameid = '0021800014'
cols = ['playerId','gameId', 'SeasonYear', 'belongTeamId','homeTeamId','awayTeamId'] 
url = 'https://tw.global.nba.com/stats2/game/preview.json?countryCode=TW&gameId={}&locale=zh_TW'.format(gameid)
url_game_data = urlopen(url)
resp = json.loads(url_game_data.read().decode('utf-8'))
hometeamid = resp['payload']['series']['games'][0]['profile']['homeTeamId']
hometeamcode = resp['payload']['homeTeam']['profile']['code']
awayteamid = resp['payload']['series']['games'][0]['profile']['awayTeamId']
awayteamcode = resp['payload']['awayTeam']['profile']['code']
gamedate = datetime.fromtimestamp(int(resp['payload']['series']['games'][0]['profile']['utcMillis'])/1000.0).date().strftime('%Y-%m-%d')
gamedate_cut = (datetime.strptime(gamedate,'%Y-%m-%d')-timedelta(days=2*365)).strftime('%Y-%m-%d')
playerid_list = list()
url = 'https://tw.global.nba.com/stats2/team/roster.json?locale=zh_TW&teamCode={}'.format(hometeamcode)
url_player_data = urlopen(url)
resp = json.loads(url_player_data.read().decode('utf-8'))
belongteamid = hometeamid
game_data = pd.DataFrame([]) 
for i in range(0,len(resp['payload']['players'])):
    playerid = resp['payload']['players'][i]['profile']['playerId']
    playerid_list.append(playerid)
    game_data_df = pd.DataFrame([playerid,gameid,gamedate,belongteamid,hometeamid, awayteamid]).transpose()
    game_data = pd.concat([game_data,game_data_df])
url = 'https://tw.global.nba.com/stats2/team/roster.json?locale=zh_TW&teamCode={}'.format(awayteamcode)
url_player_data = urlopen(url)
resp = json.loads(url_player_data.read().decode('utf-8'))
belongteamid = awayteamid
for i in range(0,len(resp['payload']['players'])):
    playerid = resp['payload']['players'][i]['profile']['playerId']
    playerid_list.append(playerid)
    game_data_df = pd.DataFrame([playerid,gameid,gamedate,belongteamid,hometeamid, awayteamid]).transpose()
    game_data = pd.concat([game_data,game_data_df])
game_data.rename(index=str, columns={0:'playerId',1:'gameId',2:'SeasonYear',3:'belongTeamId',4:'homeTeamId',5:'awayTeamId'},inplace=True)
player_data_fin = pd.DataFrame([])    
df = pd.DataFrame([])
for playerid in playerid_list:
    #SQL撈該球員近兩年資料(需要playerid,比賽日期,兩年前比賽日期)
    player_data =  pd.read_sql(
        '''SELECT * FROM nba.playerdata2
           WHERE playerId = "{}" AND SeasonYear < "{}" AND SeasonYear >= "{}"
           ORDER BY SeasonYear DESC;'''.format(playerid,gamedate,gamedate_cut),engine)
    #計算近兩年平均
    if len(player_data) > 0:
        player_data_avg = player_data.groupby(by='playerId').mean().reset_index()
        df = pd.concat([df,player_data_avg])
    career_data = pd.merge(game_data[cols],df,on=['playerId'],how='inner')
engine.dispose()

#球員近三場相同對戰系列表現
cols = ['playerId','gameId', 'SeasonYear', 'belongTeamId','homeTeamId','awayTeamId'] 
engine = create_engine("mysql+pymysql://cb102:iiicb102@10.120.23.10:3306")
engine.execute("USE nba;")
player_data_fin = pd.DataFrame([])
df = pd.DataFrame([])
for playerid in playerid_list:
    #SQL撈該球員近三場相同對戰系列資料
    player_data =  pd.read_sql(
        '''SELECT *  FROM nba.playerdata2
           WHERE playerId = "{}" AND SeasonYear < "{}" AND homeTeamId = "{}" AND awayTeamId = "{}"
           ORDER BY SeasonYear DESC
           limit 3;'''.format(playerid,gamedate,hometeamid, awayteamid),engine)
    #計算近兩年平均
    if len(player_data) > 0:
        player_data_avg = player_data.groupby(by='playerId').mean().reset_index()
        player_data_pergame = pd.merge(game_data[cols],player_data_avg,on=['playerId'],how='inner')
        df = pd.concat([df,player_data_pergame])            
    else:
        player_data_pergame = career_data[(career_data['playerId'] == playerid)].reset_index(drop=True)
        df = pd.concat([df,player_data_pergame])
    player_data_fin = df 
player_data_fin = player_data_fin.reset_index(drop=True)
engine.dispose()

#計算PIE index
def PIE_index(row):
    P_index = (row['points'] + row['fgm'] +row['ftm'] - row['fga'] - row['fta'] +row['defRebs'] + (.5 * row['offRebs']) + row['assists'] + row['steals'] + (.5 * row['blocks']) - row['fouls'] - row['turnovers']) 
    P_index = P_index
    return  P_index


def Pace(game_home,game_away):
    team_poss = game_home['fga']+0.4*game_home['fta']-1.07*((game_home['offRebs'])/(game_home['offRebs']+game_away['defRebs']))*(game_home['fga']-game_home['fgm'])+game_home['turnovers']
    opp_poss = game_away['fga']+0.4*game_away['fta']-1.07*((game_away['offRebs'])/(game_away['offRebs']+game_home['defRebs']))*(game_away['fga']-game_away['fgm'])+game_away['turnovers']
    pace = 48*(team_poss+opp_poss)/(2*((game_home['mins'])/5) )
    return pace


def Poss(game_home,game_away):
    team_poss = game_home['fga']+0.4*game_home['fta']-1.07*((game_home['offRebs'])/(game_home['offRebs']+game_away['defRebs']))*(game_home['fga']-game_home['fgm'])+game_home['turnovers']
    return team_poss

def PProd(player,game_home,game_away):
    qAST = ((player['mins'] / (game_home['mins']/5)) * (1.14*((game_home['assists']-player['assists'])/game_home['fgm'])))+(((game_home['assists']-player['assists'])/(game_home['fgm']-player['fgm']))*(1-player['mins']/((game_home['mins'])/5)))
    PProd_FG_Part = 2*(player['fgm']+0.5*player['tpm'])*(1-0.5*(player['points']-player['ftm'])/(2*player['fga'])*𝑞𝐴𝑆𝑇)
    PProd_AST_Part = 2*((game_home['fgm']-player['fgm']+0.5*(game_home['tpm']-player['tpm']))/(game_home['fgm']-player['fgm']))*0.5*((game_home['points']-game_home['ftm'])-(player['points']-player['ftm']))/(2*(game_home['fga']-player['fga']) )*player['assists']
    Team_ORB_pct = game_home['offRebs'] / (game_home['offRebs'] + game_away['rebs'] - game_away['offRebs'])
    Team_Scoring_Poss = game_home['fgm']+(1-(1-(game_home['ftm'])/(game_home['fta']))**2 )*game_home['fta']*0.4
    Team_Play_pct = Team_Scoring_Poss / (game_home['fga']+game_home['fta']*0.4+game_home['turnovers'])
    Team_ORB_Weight = ((1-Team_ORB_pct)*Team_Play_pct)/((1-Team_ORB_pct)*Team_Play_pct+Team_ORB_pct*(1-Team_Play_pct))
    PProd_ORB_Part = player['offRebs']*Team_ORB_Weight*Team_Play_pct*(game_home['points'])/(game_home['fgm']+(1-(1-(game_home['ftm'])/(game_home['fta']))**2 )*0.4*game_home['fta'])
    PProd = (PProd_FG_Part+PProd_AST_Part+player['ftm'] )*(1-(game_home['offRebs'])/(Team_Scoring_Poss)*Team_ORB_Weight*Team_Play_pct)+PProd_ORB_Part
    return PProd


def Offensive_poss(player,game_home,game_away):
    qAST = ((player['mins'] / (game_home['mins']/5)) * (1.14*((game_home['assists']-player['assists'])/game_home['fgm'])))+(((game_home['assists']-player['assists'])/(game_home['fgm']-player['fgm']))*(1-player['mins']/((game_home['mins'])/5)))
    FG_Part = player['fgm']*(1-0.5*(player['points']-player['ftm'])/(2*player['fga'])*𝑞𝐴𝑆𝑇)
    AST_Part = 0.5*((game_home['points']-game_home['ftm'])-(player['points']-player['ftm']))/(2*(game_home['fga']-player['fga']) )*player['assists']
    FT_Part = (1-(1-player['ftm']/player['fta'])**2 )*0.4*player['fta']
    Team_Scoring_Poss = game_home['fgm']+(1-(1-(game_home['ftm'])/(game_home['fta']))**2 )*game_home['fta']*0.4
    Team_ORB_pct = game_home['offRebs'] / (game_home['offRebs'] + game_away['rebs'] - game_away['offRebs'])
    Team_Play_pct = Team_Scoring_Poss / (game_home['fga']+game_home['fta']*0.4+game_home['turnovers'])
    Team_ORB_Weight = ((1-Team_ORB_pct)*Team_Play_pct)/((1-Team_ORB_pct)*Team_Play_pct+Team_ORB_pct*(1-Team_Play_pct))
    ORB_Part =player['offRebs']*Team_ORB_Weight*Team_Play_pct
    ScPoss = (FG_Part+AST_Part+FT_Part)*(1-game_home['offRebs'] /Team_Scoring_Poss *Team_ORB_Weight *Team_Play_pct)+ORB_Part
    FGxPoss =(player['fga']-player['fgm'])*(1-1.07*Team_ORB_pct)
    FTxPoss = (1-(player['ftm']/player['fta'])**2 )*0.4*player['fta']
    offensive_poss = ScPoss + FGxPoss + FTxPoss + player['turnovers']
    return offensive_poss
    

def DRtg(player, game_home, game_away):
    D_Pts_per_ScPoss = (game_away['points'])/(game_away['fgm']+(1-(1-(game_away['ftm']/game_away['fta'])**2 ))*game_away['fta']*0.4)
    Team_Defensive_Rating = 100 * game_away['points'] / Poss(game_home,game_away)
    DOR_pct =  game_away['points'] / ( game_away['offRebs'] + game_home['defRebs']) 
    DFG_pct = game_away['fgm'] / game_away['fga'] 
    FMwt =  (DFG_pct * (1 - DOR_pct) )/(DFG_pct * (1 - DOR_pct) + (1 - DFG_pct) * DOR_pct )
    Stops1 = player['steals'] + player['blocks'] * FMwt *(1 - 1.07 * DOR_pct) + player['defRebs'] * (1 - FMwt)
    Stops2 = (game_away['fga']-game_away['fgm']-game_home['blocks']) * FMwt * (1 - 1.07 * DOR_pct) /(game_home['mins']) +(game_away['turnovers']-game_home['steals'])/(game_home['mins'])*player['mins'] + player['fouls']/(game_home['fouls'])*0.4*game_away['fta']*(1-(game_away['ftm']/game_away['fta'])**2 )
    Stop_pct = Stops1 + Stops2
    DRtg = Team_Defensive_Rating + 0.2 * (100 * D_Pts_per_ScPoss * (1 - Stop_pct) - Team_Defensive_Rating)    
    return DRtg


player_data = player_data_fin
team_data = player_data.groupby(by=['gameId','belongTeamId']).sum().reset_index()
Lg_avg = pd.DataFrame(career_data[['assists','blocks','defRebs', 'fga', 'fgm', 'fouls', 'fta', 'ftm','mins', 'offRebs','points', 'rebs', 'steals', 'tpa', 'tpm','turnovers']].mean(0)).transpose()
Lg_avg = Lg_avg.rename(index=str,columns={'assists':'assistsPg','blocks':'blocksPg','defRebs':'defRebsPg', 'fga':'fgaPg', 'fgm':'fgmPg', 'fouls':'foulsPg', 'fta':'ftaPg', 'ftm':'ftmPg','mins':'minsPg', 'offRebs':'offRebsPg','points':'pointsPg', 'rebs':'rebsPg', 'steals':'stealsPg', 'tpa':'tpaPg', 'tpm':'tpmPg','turnovers':'turnoversPg'})


#計算聯盟pace%pts per poss
poss_df = pd.DataFrame([])
pace_df = pd.DataFrame([])
game_data_team = team_data[team_data['gameId']== gameid].reset_index(drop=True)
game_home = game_data_team[0:1].reset_index(drop=True)
game_away = game_data_team[1:2].reset_index(drop=True)
poss_df_ = pd.DataFrame([Poss(game_home,game_away),Poss(game_away,game_home)])
pace_df_ = pd.DataFrame([Pace(game_home,game_away),Pace(game_away,game_home)])
poss_df = pd.concat([poss_df,poss_df_])
pace_df = pd.concat([pace_df,pace_df_])
lg_pace = pace_df.mean()[0]
lg_total_poss = poss_df.sum()[0]
lg_total_pts = team_data.sum()['points']
lg_points_per_poss = lg_total_pts / lg_total_poss


record_index_df = pd.DataFrame([])
game_data_sum = team_data
record_index_df_p = pd.DataFrame([])
for playerid in playerid_list:
    player = player_data[player_data['playerId'] == playerid].reset_index(drop=True)
    if len(player) > 0:
        if game_data_sum['belongTeamId'][0] == player['belongTeamId'][0]:        
            game_home = game_data_sum[0:1].reset_index(drop=True)
            game_away = game_data_sum[1:2].reset_index(drop=True)
        else:
            game_away = game_data_sum[0:1].reset_index(drop=True)
            game_home = game_data_sum[1:2].reset_index(drop=True)
        Marginal_pts_per_win = 0.32 * Lg_avg['pointsPg'][0]*(Pace(game_home,game_away)[0]/lg_pace)
        Marginal_Offense = PProd(player,game_home,game_away)[0]-0.92*(lg_points_per_poss)*(Offensive_poss(player,game_home,game_away)[0])
        Marginal_Defense = player['mins'][0] /game_home['mins'][0]  * Poss(game_away,game_home)[0] *(1.08*lg_points_per_poss-DRtg(player, game_home, game_away)[0] /100)
        OWS = Marginal_Offense / Marginal_pts_per_win
        DWS = Marginal_Defense / Marginal_pts_per_win
        WS = OWS + DWS
        df_ = pd.DataFrame([gameid,playerid,player['belongTeamId'][0],Pace(game_home,game_away)[0],PProd(player,game_home,game_away)[0],Offensive_poss(player,game_home,game_away)[0],
                            DRtg(player, game_home, game_away)[0],OWS,DWS,WS]).transpose()
        record_index_df_p =  pd.concat([record_index_df_p,df_])
record_index_df_p = record_index_df_p.dropna()
record_index_df =  pd.concat([record_index_df,record_index_df_p])
record_index_df.rename(index=str, columns={0:"gameId", 1:"playerId", 2:"belongTeamId", 3:"Pace", 4:"PProd", 5:"Offensive_poss", 6:"DRtg", 7:"OWS", 8:"DWS", 9:"WS"},inplace=True)
record_index_df = record_index_df.reset_index(drop=True)

from bs4 import BeautifulSoup
import requests

#Gambling data crawl
headers = ['對戰組合', '比數', '時間', '初盤', '看好度', '台灣運彩', '九州娛樂城', 'Pinnacle', 'Bookmaker', 'bet365', 'BetOnline']
df = pd.DataFrame([])
start_dt = '20181010'
start_dt_parse = datetime.strptime(start_dt,"%Y%m%d").date()
url_game_dt = datetime.strftime((start_dt_parse -  timedelta(days=0)),"%Y%m%d")
url = 'https://www.lottonavi.com/odds/nba/{}/'.format(url_game_dt)
res = requests.get(url)
soup = BeautifulSoup(res.content,'lxml')
#季賽名稱
game_season = soup.select('.breadcrumb li')[2].text
#GAME DATE
game_dt = url_game_dt
#Gambling
df_ = pd.read_html(soup.select('table')[0].prettify())[0]
if len(df_) != 0:
    df_.columns = headers
    gt = [game_dt]*len(df_)
    df_['日期'] = gt
    df_['季賽'] = game_season
    df = pd.concat([df,df_])
df.reset_index(drop=True)
#連入資料庫抓team_info TABLE資料
engine = create_engine("mysql+pymysql://cb102:iiicb102@10.120.23.10:3306")
engine.execute("USE nba;")
team_info = pd.read_sql("select * from team_info;",engine)
engine.dispose()
#抓取隊伍名稱&ID
team_dict = dict()
for i in range(0,len(team_info['name'])):
    team_dict[team_info['name'][i]] = team_info['teamId'][i]
team_dict['塞爾提克'] = '1610612738'
team_dict['七六人'] = '1610612755'
team_dict['黃蜂'] = '1610612766'


#games資料取'gameId','gameTime','awayTeamID','homeTeamID'這幾個欄位
games = game_data[['gameId','SeasonYear','awayTeamId','homeTeamId']].drop_duplicates()

#清理gambling & 比對gameID
df_gambling = pd.DataFrame([])
df_ = dict()
for i in range(0,len(df['對戰組合'])):
    if not pd.isna(df['台灣運彩'][i]):
        df_['SeasonYear'] = games['SeasonYear'][0]
        if len(df['對戰組合'][i].split('  ')) == 2:
            df_['away'] = df['對戰組合'][i].split('  ')[0]
            df_['home'] = df['對戰組合'][i].split('  ')[1]
        elif len(df['對戰組合'][i].split('  ')) == 3:
            if re.match('\\(',df['對戰組合'][i].split('  ')[1]):
                df_['away'] = df['對戰組合'][i].split('  ')[0]
                df_['home'] = df['對戰組合'][i].split('  ')[2]
            else:
                df_['away'] = df['對戰組合'][i].split('  ')[0]
                df_['home'] = df['對戰組合'][i].split('  ')[1]                           
        else:
            df_['away'] = df['對戰組合'][i].split('  ')[0]
            df_['home'] = df['對戰組合'][i].split('  ')[2]
        df_['homeTeamId'] = team_dict[ df_['home']]
        df_['awayTeamId'] = team_dict[ df_['away']]
        if float(df['台灣運彩'][i].split('  ')[0]) > 100:
            df_['spread'] = float(df['台灣運彩'][i].split('  ')[1])
            df_['totalPoint'] = float(df['台灣運彩'][i].split('  ')[0])
        else:
            df_['spread'] = -float(df['台灣運彩'][i].split('  ')[0])
            df_['totalPoint'] = float(df['台灣運彩'][i].split('  ')[1])
        df_temp = pd.DataFrame(df_,index=[0])
        df_gambling = pd.concat([df_gambling,df_temp])
df_gambling = df_gambling.reset_index(drop=True)    
gambling_data = pd.merge(games, df_gambling, how = 'left', left_on = ['SeasonYear','homeTeamId','awayTeamId'], right_on =  ['SeasonYear','homeTeamId','awayTeamId'])


#Data combime & ETL for TotalPoints
player_data = player_data_fin 
model_data_cols = ['gameId', 'assists', 'blocks','defRebs', 'fga', 'fgm', 'fouls', 'fta', 'ftm','mins', 'offRebs', 'points', 
                    'rebs', 'steals', 'tpa', 'tpm','turnovers']
model_data_x = player_data[model_data_cols].groupby(by='gameId').sum().reset_index()
model_data_x1 = pd.merge(gambling_data,model_data_x,on='gameId',how='inner')
model_data_x1['PIE'] = model_data_x1.apply(PIE_index,axis=1)
model_data = pd.merge(model_data_x1,record_index_df.groupby(by=['gameId']).sum().reset_index(),on='gameId',how='inner')
model_data['SeasonYear'] = model_data['SeasonYear'].astype(str)
model_data_filter = model_data


import pickle


#Load model
with open('./Logistic_TotalPoint.pickle', 'rb') as f:
    Logistic_TotalPoint = pickle.load(f)
with open('./Logistic_gamblingWin.pickle', 'rb') as f:
    Logistic_gamblingWin = pickle.load(f)
with open('./Logistic_gameWin.pickle', 'rb') as f:
    Logistic_gameWin = pickle.load(f)


model_x_cols = ['PIE','PProd','OWS']
model_data_x = model_data_filter[model_x_cols]
print(Logistic_TotalPoint.predict_proba(model_data_x))
if(Logistic_TotalPoint.predict_proba(model_data_x)[0][0]>Logistic_TotalPoint.predict_proba(model_data_x)[0][1]):
    print('Total points > {} '.format(model_data['totalPoint'][0]))
else:
     print('Total points < {} '.format(model_data['totalPoint'][0]))


model_x_cols = ['PIE','Pace','PProd','OWS']
model_data_x = model_data_filter[model_x_cols]
print(Logistic_gamblingWin.predict_proba(model_data_x))
if(Logistic_gamblingWin.predict_proba(model_data_x)[0][0]>Logistic_gamblingWin.predict_proba(model_data_x)[0][1]):
    print('讓分:{} Win'.format(model_data['home'][0]))
else:
    print('讓分: {} Win'.format(model_data['away'][0]))


model_x_cols = ['PIE','Pace','PProd','OWS']
model_data_x = model_data_filter[model_x_cols]
print(Logistic_gameWin.predict_proba(model_data_x))
if(Logistic_gameWin.predict_proba(model_data_x)[0][0]>Logistic_gameWin.predict_proba(model_data_x)[0][1]):
    print('不讓分:{} Win'.format(model_data['home'][0]))
else:
    print('不讓分: {} Win'.format(model_data['away'][0]))

