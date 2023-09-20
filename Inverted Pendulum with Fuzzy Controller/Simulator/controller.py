# -*- coding: utf-8 -*-

# python imports
from math import degrees, e

# import numpy as np

# pyfuzzy imports
from fuzzy.storage.fcl.Reader import Reader


class FuzzyController:

    def __init__(self,fcl_path):
        # self.system = Reader().load_from_file(fcl_path)
        self.pa_array=[
            ("up_more_right",(0,30,60)),
            ("up_right",(30,60,90)),
            ("up",(60,90,120)),
            ("up_left",(90,120,150)),
            ("up_more_left",(120,150,180)),
            ("down_more_left",(180,210,240)),
            ("down_left",(210,240,270)),
            ("down",(240,270,300)),
            ("down_right",(270,300,330)),
            ("down_more_right",(300,330,360))
        ]
        self.pv_array=[
            ("cw_fast",(-200,-100)),
            ("cw_slow",(-200,-100,0)),
            ("stop",(-100,0,100)),
            ("ccw_slow",(0,100,200)),
            ("ccw_fast",(100,200))
        ]
        self.force_array=[
            ("left_fast",(-100,-80,-60)),
            ("left_slow",(-80,-60,0)),
            ("stop",(-60,0,60)),
            ("right_slow",(0,60,80)),
            ("right_fast",(60,80,100))
        ]
        self.rules_array=[
            ( "up_more_right" , "ccw_slow" , "right_fast"),
            ( "up_more_right" , "cw_slow" , "right_fast"),
            ( "up_more_left" , "cw_slow" , "left_fast"),
            ( "up_more_left" , "ccw_slow" , "left_fast"),
            ( "up_more_right" , "ccw_fast" , "left_slow"),
            ( "up_more_right" , "cw_fast" , "right_fast"),
            ( "up_more_left" , "cw_fast" , "right_slow"),
            ( "up_more_left" , "ccw_fast" , "left_fast"),
            ( "down_more_right" , "ccw_slow" , "right_fast"),
            ( "down_more_right" , "cw_slow" , "stop"),
            ( "down_more_left" , "cw_slow" , "left_fast"),
            ( "down_more_left" , "ccw_slow" , "stop"),
            ( "down_more_right" , "ccw_fast" , "stop"),
            ( "down_more_right" , "cw_fast" , "stop"),
            ( "down_more_left" , "cw_fast" , "stop"),
            ( "down_more_left" , "ccw_fast" , "stop"),
            ( "down_right" , "ccw_slow" , "right_fast"),
            ( "down_right" , "cw_slow" , "right_fast"),
            ( "down_left" , "cw_slow" , "left_fast"),
            ( "down_left" , "ccw_slow" , "left_fast"),
            ( "down_right" , "ccw_fast" , "stop"),
            ( "down_right" , "cw_fast" , "right_slow"),
            ( "down_left" , "cw_fast" , "stop"),
            ( "down_left" , "ccw_fast" , "left_slow"),
            ( "up_right" , "ccw_slow" , "right_slow"),
            ( "up_right" , "cw_slow" , "right_fast"),
            ( "up_right" , "stop" , "right_fast"),
            ( "up_left" , "cw_slow" , "left_slow"),
            ( "up_left" , "ccw_slow" , "left_fast"),
            ( "up_left" , "stop" , "left_fast"),
            ( "up_right" , "ccw_fast" , "left_fast"),
            ( "up_right" , "cw_fast" , "right_fast"),
            ( "up_left" , "cw_fast" , "right_fast"),
            ( "up_left" , "ccw_fast" , "left_fast"),
            ( "down" , "stop" , "right_fast"),
            ( "down" , "cw_fast" , "stop"),
            ( "down" , "ccw_fast" , "stop"),
            ( "up" , "ccw_slow" , "left_slow"),
            ( "up" , "ccw_fast" , "left_fast"),
            ( "up" , "cw_slow" , "right_slow"),
            ( "up" , "cw_fast" , "right_fast"),
            ( "up" , "stop" , "stop")
        ]
        self.rules_amount={
            "left_fast":0.0,
            "left_slow":0.0,
            "stop":0.0,
            "right_slow":0.0,
            "right_fast":0.0
        }

    def reset_rules_amount(self):
        """
        Reset rules amount before new force
        """
        self.rules_amount={
            "left_fast":0.0,
            "left_slow":0.0,
            "stop":0.0,
            "right_slow":0.0,
            "right_fast":0.0
        }

    def linspace(self,st,en,num):
        """
        a method for spliting a range into num numbers
        """
        step=1.0*(float(en)-float(st))/float(num)
        a=[]
        i=st
        while i<=en:
            a.append(i)
            i+=step
        return a
        

    def m_and_h_calculator(self,x1,y1,x2,y2):
        """
        a method for calculating m and h in mx+h
        """
        m=1.0*(float(y2)-float(y1))/(float(x2)-float(x1))
        h=float(y1)-float(m)*float(x1)
        return (m,h)

    def fuzzify_pv(self,x):
        """"
        a method for fuzzifying pv
        """
        answer={}
        for pv in self.pv_array:
            u=0
            name=pv[0]
            if name=="cw_fast" or name=="ccw_fast":
                startX=pv[1][0]
                endX=pv[1][1]
                if x<=endX and x>=startX:
                    m=0
                    h=0
                    if name=="cw_fast":
                        m,h=self.m_and_h_calculator(startX,1,endX,0)
                    elif name=="ccw_fast":
                        m,h=self.m_and_h_calculator(startX,0,endX,1)
                    u=float(m)*float(x)+float(h)
                elif name=="cw_fast" and x<startX:
                    u=1
                elif name=="ccw_fast" and x>endX:
                    u=1
                else:
                    u=0
                answer[name]=u
            else:
                startX=pv[1][0]
                middleX=pv[1][1]
                endX=pv[1][2]
                if x<=middleX and x>=startX:
                    m,h=self.m_and_h_calculator(startX,0,middleX,1)
                    u=float(m)*float(x)+float(h)
                elif x>=middleX and x<=endX:
                    m,h=self.m_and_h_calculator(middleX,1,endX,0)
                    u=float(m)*float(x)+float(h)
                else:
                    u=0
                answer[name]=u
        return answer

    def fuzzify_pa(self,x):
        """"
        a method for fuzzifying pa
        """
        answer={}
        for pa in self.pa_array:
            u=0
            name=pa[0]
            startX=pa[1][0]
            middleX=pa[1][1]
            endX=pa[1][2]
            if x<=middleX and x>=startX:
                m,h=self.m_and_h_calculator(startX,0,middleX,1)
                u=float(m)*float(x)+float(h)
            elif x>=middleX and x<=endX:
                m,h=self.m_and_h_calculator(middleX,1,endX,0)
                u=float(m)*float(x)+float(h)
            else:
                u=0
            answer[name]=u
        return answer

    def fuzzify(self,pa,pv):
        """
        a method for fuzzifying pa and pv
        """
        pa_amount=self.fuzzify_pa(pa)
        pv_amount=self.fuzzify_pv(pv)
        return (pa_amount,pv_amount)

    def inference(self,pa_amount,pv_amount):
        """
        a method for inferencing from pa and pv
        """
        # rule 0:
        self.rules_amount["stop"]=max(min(pa_amount["up"],pv_amount["stop"]),min(pa_amount["up_right"],pv_amount["ccw_slow"]),min(pa_amount["up_left"],pv_amount["cw_slow"]))
        # other rules :
        for rule in self.rules_array:
            tmp=min(pa_amount[rule[0]],pv_amount[rule[1]])
            if tmp>self.rules_amount[rule[2]]:
                self.rules_amount[rule[2]]=tmp

    def max_force(self,x):
        """
        finding max force for integral
        """
        max_u=0.0
        for force in self.force_array:
            u=0.0
            name=force[0]
            if self.rules_amount[name]==0:
                continue
            startX=force[1][0]
            middleX=force[1][1]
            endX=force[1][2]
            if x<=middleX and x>=startX:
                m,h=self.m_and_h_calculator(startX,0,middleX,1)
                u=float(m)*float(x)+float(h)
                if u>self.rules_amount[name]:
                    u=self.rules_amount[name]
            elif x>=middleX and x<=endX:
                m,h=self.m_and_h_calculator(middleX,1,endX,0)
                u=float(m)*float(x)+float(h)
                if u>self.rules_amount[name]:
                    u=self.rules_amount[name]
            else:
                u=0.0
            
            if u>max_u:
                max_u=u
        return max_u
            
        

    def debug_rules(self):
        """
        debug exceptions
        """
        for key in self.rules_amount.keys():
            if self.rules_amount[key]!=0:
                return
        print("okb")
        quit()

    def defuzzify(self):
        """
        a method for defuzzifying using integral
        """
        # self.debug_rules()
        soorat=0.0
        makhraj=0.0
        X=self.linspace(-100,100,1000)
        delta=X[1]-X[0]
        for i in X:
            U=self.max_force(i)
            soorat+=U*i*delta
            makhraj+=U*delta
        center=0.0
        if makhraj!=0:
            center=1.0*float(soorat)/float(makhraj)

        return center

        

    def _make_input(self, world):
        return dict(
            cp = world.x,
            cv = world.v,
            pa = degrees(world.theta),
            pv = degrees(world.omega)
        )




    def _make_output(self):
        return dict(
            force = 0.
        )


    def decide(self, world):
        """
        main method for doin all the phases and returning the final answer for force
        """
        input = self._make_input(world)
        self.reset_rules_amount()
        pa_amount,pv_amount=self.fuzzify(float(input["pa"]),float(input["pv"]))
        self.inference(pa_amount,pv_amount)
        first=self.defuzzify()
        return first
        
