import React, { useState } from 'react';
import { Heart, Phone, MapPin, Calendar, ChevronRight, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';

// Mock family data
const familyMembers = [
  {
    id: 1,
    name: 'Maria',
    relationship: 'Spouse',
    relationshipLabel: 'Your Wife',
    photo: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&h=400&fit=crop&crop=face',
    phone: '(555) 123-4567',
    address: '123 Oak Street, Hometown',
    birthday: 'March 15',
    yearsMarried: 52,
    favoriteMemory: 'Your wedding day in 1972 at the little church on Maple Avenue. Maria wore her mother\'s wedding dress.',
    category: 'spouse',
    color: 'border-family-spouse bg-family-spouse/10'
  },
  {
    id: 2,
    name: 'Michael',
    relationship: 'Son',
    relationshipLabel: 'Your Son',
    photo: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face',
    phone: '(555) 234-5678',
    address: '456 Pine Road, Nearby City',
    birthday: 'July 22',
    favoriteMemory: 'Teaching him to ride a bike in the backyard when he was 6. He fell many times but never gave up!',
    category: 'children',
    color: 'border-family-children bg-family-children/10'
  },
  {
    id: 3,
    name: 'Sarah',
    relationship: 'Daughter',
    relationshipLabel: 'Your Daughter',
    photo: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=400&fit=crop&crop=face',
    phone: '(555) 345-6789',
    address: '789 Elm Avenue, Sunny Town',
    birthday: 'November 8',
    favoriteMemory: 'Her graduation day from medical school. You were so proud, you cried happy tears.',
    category: 'children',
    color: 'border-family-children bg-family-children/10'
  },
  {
    id: 4,
    name: 'Emma',
    relationship: 'Granddaughter',
    relationshipLabel: 'Your Granddaughter',
    photo: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?w=400&h=400&fit=crop&crop=face',
    phone: '(555) 456-7890',
    address: 'Lives with Michael',
    birthday: 'April 3',
    age: 12,
    favoriteMemory: 'She loves baking cookies with you every Christmas. Her favorite are your chocolate chip cookies.',
    category: 'grandchildren',
    color: 'border-family-grandchildren bg-family-grandchildren/10'
  },
  {
    id: 5,
    name: 'James',
    relationship: 'Grandson',
    relationshipLabel: 'Your Grandson',
    photo: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400&h=400&fit=crop&crop=face',
    phone: '(555) 567-8901',
    address: 'Lives with Sarah',
    birthday: 'September 17',
    age: 8,
    favoriteMemory: 'He calls you "Papa Bear" because you always give the best bear hugs. He loves fishing with you.',
    category: 'grandchildren',
    color: 'border-family-grandchildren bg-family-grandchildren/10'
  },
  {
    id: 6,
    name: 'Lily',
    relationship: 'Granddaughter',
    relationshipLabel: 'Your Granddaughter',
    photo: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&h=400&fit=crop&crop=face',
    phone: '(555) 678-9012',
    address: 'Lives with Sarah',
    birthday: 'February 28',
    age: 5,
    favoriteMemory: 'The youngest! She loves when you read her bedtime stories, especially "The Little Prince".',
    category: 'grandchildren',
    color: 'border-family-grandchildren bg-family-grandchildren/10'
  },
];

const categoryLabels = {
  spouse: { label: 'My Spouse', color: 'bg-family-spouse text-white' },
  children: { label: 'My Children', color: 'bg-family-children text-white' },
  grandchildren: { label: 'My Grandchildren', color: 'bg-family-grandchildren text-foreground' },
};

export const FamilySection = () => {
  const [selectedMember, setSelectedMember] = useState(null);
  const [filter, setFilter] = useState('all');

  const filteredMembers = filter === 'all' 
    ? familyMembers 
    : familyMembers.filter(m => m.category === filter);

  const groupedMembers = {
    spouse: filteredMembers.filter(m => m.category === 'spouse'),
    children: filteredMembers.filter(m => m.category === 'children'),
    grandchildren: filteredMembers.filter(m => m.category === 'grandchildren'),
  };

  return (
    <section className="py-8 sm:py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-2 mb-4">
            <Heart className="h-5 w-5 text-primary" fill="currentColor" />
            <span className="text-base font-medium text-primary">Your Loved Ones</span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold text-foreground mb-4">
            My Family
          </h2>
          <p className="text-accessible text-muted-foreground max-w-2xl mx-auto">
            Tap on anyone to learn more about them
          </p>
        </div>

        {/* Filter buttons */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
          <Button
            variant={filter === 'all' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('all')}
          >
            Everyone
          </Button>
          <Button
            variant={filter === 'spouse' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('spouse')}
            className={filter === 'spouse' ? 'bg-family-spouse hover:bg-family-spouse/90' : ''}
          >
            My Spouse
          </Button>
          <Button
            variant={filter === 'children' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('children')}
            className={filter === 'children' ? 'bg-family-children hover:bg-family-children/90' : ''}
          >
            My Children
          </Button>
          <Button
            variant={filter === 'grandchildren' ? 'accessible' : 'accessible-outline'}
            onClick={() => setFilter('grandchildren')}
            className={filter === 'grandchildren' ? 'bg-family-grandchildren hover:bg-family-grandchildren/90 text-foreground' : ''}
          >
            My Grandchildren
          </Button>
        </div>

        {/* Family members by category */}
        {Object.entries(groupedMembers).map(([category, members]) => {
          if (members.length === 0) return null;
          const categoryInfo = categoryLabels[category];
          
          return (
            <div key={category} className="mb-12">
              <div className="flex items-center gap-3 mb-6">
                <Badge className={`${categoryInfo.color} text-lg px-4 py-2`}>
                  {categoryInfo.label}
                </Badge>
                <div className="flex-1 h-px bg-border" />
              </div>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {members.map((member, index) => (
                  <Card
                    key={member.id}
                    className={`group cursor-pointer border-2 ${member.color} shadow-card hover:shadow-elevated transition-all duration-300 hover:-translate-y-1 animate-scale-in overflow-hidden`}
                    style={{ animationDelay: `${index * 0.1}s` }}
                    onClick={() => setSelectedMember(member)}
                  >
                    <CardContent className="p-0">
                      {/* Photo */}
                      <div className="relative h-48 sm:h-56 overflow-hidden">
                        <img
                          src={member.photo}
                          alt={member.name}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
                      </div>
                      
                      {/* Info */}
                      <div className="p-6">
                        <h3 className="text-2xl font-bold text-foreground mb-1">{member.name}</h3>
                        <p className="text-lg text-primary font-medium mb-3">{member.relationshipLabel}</p>
                        
                        <div className="flex items-center gap-2 text-muted-foreground mb-4">
                          <Calendar className="h-5 w-5" />
                          <span className="text-base">Birthday: {member.birthday}</span>
                        </div>
                        
                        <Button variant="outline" className="w-full gap-2 group-hover:bg-primary group-hover:text-primary-foreground">
                          <span>Learn More</span>
                          <ChevronRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          );
        })}

        {/* Detail Dialog */}
        <Dialog open={!!selectedMember} onOpenChange={() => setSelectedMember(null)}>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            {selectedMember && (
              <>
                <DialogHeader>
                  <DialogTitle className="text-3xl font-display">{selectedMember.name}</DialogTitle>
                  <DialogDescription className="text-xl text-primary font-medium">
                    {selectedMember.relationshipLabel}
                  </DialogDescription>
                </DialogHeader>
                
                <div className="mt-6 space-y-6">
                  {/* Large Photo */}
                  <div className="relative rounded-2xl overflow-hidden shadow-card">
                    <img
                      src={selectedMember.photo}
                      alt={selectedMember.name}
                      className="w-full h-64 sm:h-80 object-cover"
                    />
                  </div>
                  
                  {/* Details */}
                  <div className="space-y-4">
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                      <Calendar className="h-6 w-6 text-primary mt-1" />
                      <div>
                        <p className="font-semibold text-lg">Birthday</p>
                        <p className="text-accessible text-muted-foreground">{selectedMember.birthday}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                      <Phone className="h-6 w-6 text-primary mt-1" />
                      <div>
                        <p className="font-semibold text-lg">Phone Number</p>
                        <p className="text-accessible text-muted-foreground">{selectedMember.phone}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-muted">
                      <MapPin className="h-6 w-6 text-primary mt-1" />
                      <div>
                        <p className="font-semibold text-lg">Address</p>
                        <p className="text-accessible text-muted-foreground">{selectedMember.address}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-start gap-4 p-4 rounded-xl bg-primary/10 border-2 border-primary">
                      <Heart className="h-6 w-6 text-primary mt-1" fill="currentColor" />
                      <div>
                        <p className="font-semibold text-lg text-primary">A Special Memory</p>
                        <p className="text-accessible text-foreground mt-2">{selectedMember.favoriteMemory}</p>
                      </div>
                    </div>
                  </div>
                  
                  <Button 
                    variant="accessible" 
                    className="w-full"
                    onClick={() => setSelectedMember(null)}
                  >
                    <X className="h-6 w-6 mr-2" />
                    Close
                  </Button>
                </div>
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </section>
  );
};

export default FamilySection;
