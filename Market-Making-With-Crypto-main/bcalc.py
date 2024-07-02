import argparse
import sys
# from errors import Error
import numpy as np
import pandas as pd

class bond:
    def bondc(face, coupon, maturity, discount):
        discounted_final_cf = (face + (coupon*face))/(1+discount)**maturity
        dmac = discounted_final_cf * maturity    
        maturity -= 1  
        discounted_cf = 0
        
        while maturity > 0:
            discounted_cf = (coupon*face) + (discounted_cf/(1+discount)**maturity)
            dmac = dmac + discounted_cf
            maturity -= 1 
            
        price = discounted_cf + discounted_final_cf
        dmac = dmac / price
        dmod = dmac / (1+discount)
        
        mv = price/100 * face
        
        #      price, dmac, dmod, Dol Dura,  DV01
        return price, dmac, dmod, mv*dmod,  ((mv*dmod) * 0.01)
    
    # FrbDirtyPriceFromYieldCHN(BondObject,Yield,isCompounding)
    def price(coupon, maturity):
        return bond.bondc(100, coupon, maturity, 0.0)[0];
    
    # FrbDurationCHN(BondObject,Yield)
    def duration(coupon, maturity):
        return bond.bondc(100, coupon, maturity, 0.0)[1];
    
    # FrbMDurationCHN(CBondObject,Yield)
    def dmad(coupon, maturity):
        return bond.bondc(100, coupon, maturity, 0.0)[2];
    
    # FrbPVBPCHN(CBondObject,Yield)
    def dv01(coupon, maturity):
        return bond.bondc(100, coupon, maturity, 0.0)[4];


class Indicator:

    # def __init__(self, data):
    #     self.data = np.array(data[0])
    #     self.df = data[0]

    def final_value(ts):
        return ts[-1]

    def MaxDrawdown(ts):
        index_j = np.argmax(np.maximum.accumulate(ts) - ts)  # 结束位置
        if index_j == 0:
            return 0
        index_i = np.argmax(ts[:index_j])  # 开始位置
        d = (ts[index_i] - ts[index_j])  # 最大回撤
        return d
    
    def MaxProfit(ts):
        index_j = np.argmax(ts - np.minimum.accumulate(ts))  # 结束位置
        if index_j == 0:
            return 0
        index_i = np.argmin(ts[:index_j])  # 开始位置
        d = (ts[index_j] - ts[index_i]) # 最大赢率
        return d
    
    def StdDev(ts):
        stdp = np.std(np.maximum.accumulate(ts))
        return stdp
    def Average(ts):
        return np.mean(ts)
    
    def Median(ts):
        return np.median(ts[~np.isnan(ts)])
    
    def Range(ts):
        return np.max(ts) - np.min(ts)

    def sharpe_ratio(ts):
        '''夏普比率'''
        returns = ts.shift(1) - ts.shift(0)  # 每日收益
        average_return = np.mean(returns)
        return_stdev = np.std(returns)

        AnnualRet = average_return * 252  # 默认252个工作日
        AnnualVol = return_stdev * np.sqrt(252)
        sharpe_ratio = (AnnualRet - 0.02) / AnnualVol  # 默认无风险利率为0.02
        return (sharpe_ratio)
    
    
    
# if __name__ == '__main__':
    
#     argparser = argparse.ArgumentParser(description='Simple Bond Calculator')
#     argparser.add_argument('--face', dest='face', type=float, default='100.00', help='Face value of bond (default $100.00)')
#     argparser.add_argument('--coupon', dest='coupon', type=float, help='Coupon rate (Annual)')
#     argparser.add_argument('--maturity', dest='maturity', type=int, help='Years to maturity')
#     argparser.add_argument('--discount', dest='discount', type=float, help='Discount Rate (Annual)')
#     argparser.add_argument('--position', dest='position', type=float, help='Number of bonds in position')
#     args = argparser.parse_args()
    
#     #TODO: Convert to semi annual payments 
    
#     # Validate Input 
#     # if float(args.coupon) > 1 or float(args.coupon) < 0:
#     #     raise Error ('Coupon must be between 0 and 1')
    
#     # if args.discount > 1 or args.coupon < 0:
#     #     raise Error ('Coupon must be between 0 and 1')
        
#     # Figure out what needs to be calculated 
#     # try:
#     price, dmac, dmod = calculate(args.face, args.coupon, args.maturity, args.discount) 
#     mv = (args.position * price/100 * args.face)
#     # except TypeError:
#     #     raise Error ('Error in input, please check')
    
#     # Output 
#     print('Simple Bond Calculator') 
#     print('Face: $%s' % args.face) 
#     print('Coupon (Annual): $%s' % (args.coupon*args.face)) 
#     print('Price: $%s' % price) 
#     print('Maturity: %s years' % args.maturity) 
#     print('Discount: %s' % args.discount) 
#     print('DMac: %s' %dmac) 
#     print('DMod: %s' %dmod) 
#     print('Market Value: %s' % mv)   
#     print('Dollar Duration: %s' % (mv*dmod))     
#     print('DV01: %s' % ((mv*dmod) * 0.01)) 
    
    